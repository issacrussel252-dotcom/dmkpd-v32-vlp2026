from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v31_core import (BASE, BASE_SHA, GRID, HEAD_SHA, MAX_CANDIDATES, ROOT, SEED,
                      CornerResidualHead, append_jsonl, atomic_torch_save,
                      capture_rng, decode_flat_candidate_rows, dump_json, load_models, restore_rng,
                      safe_torch_load, seed_all, set_v28_backend, sha256)
from v31_core import roi_corner_coordinates
from v9_train import state_digest


EPOCHS = 3
MICROBATCH = 256
ACCUMULATION = 2
EFFECTIVE_BATCH = MICROBATCH * ACCUMULATION
LR = 1e-4
WEIGHT_DECAY = 1e-4
BACKGROUND_WEIGHT = 0.1
CHECKPOINT_EVERY_STEPS = 100


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def load_train_maps():
    cache = ROOT / "cache" / "train_fit"
    complete = json.loads((cache / "collection_complete.json").read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("role") != "train_fit":
        raise RuntimeError("full train_fit feature collection is not complete")
    features = np.load(cache / "features.npy", mmap_mode="r")
    names = ("candidate_ids", "group", "target_norm", "corner_norm", "rois_resized",
             "candidate_count", "matched_small", "matched_tiny", "raw_v28", "image_size", "resized_size")
    maps = {n: np.load(cache / f"{n}.npy", mmap_mode="r") for n in names}
    if features.shape[0] != 7731 or features.shape[1] != MAX_CANDIDATES or features.shape[2:] != (256, 4, 4):
        raise RuntimeError(f"unexpected train cache feature shape: {features.shape}")
    return features, maps


def make_epoch_order(maps, epoch: int):
    rng = np.random.default_rng(SEED + 10007 * epoch)
    image_order = rng.permutation(len(maps["candidate_count"]))
    selected: list[int] = []
    coverage = {"images_seen": 0, "images_with_foreground": 0, "images_with_background": 0,
                "foreground_selected": 0, "background_pool": 0, "background_selected": 0,
                "ignored_not_selected": 0}
    group_map = maps["group"]
    for image_idx in image_order:
        count = int(maps["candidate_count"][image_idx, 0])
        groups = np.asarray(group_map[image_idx, :count])
        fg = np.flatnonzero(groups == 1)
        bg = np.flatnonzero(groups == 2)
        coverage["images_seen"] += 1
        coverage["images_with_foreground"] += int(len(fg) > 0)
        coverage["images_with_background"] += int(len(bg) > 0)
        coverage["foreground_selected"] += len(fg)
        coverage["background_pool"] += len(bg)
        bg_n = min(len(bg), len(fg))
        if bg_n:
            bg = rng.choice(bg, size=bg_n, replace=False)
        else:
            bg = np.empty(0, dtype=np.int64)
        local = np.concatenate((fg.astype(np.int64), bg.astype(np.int64)))
        rng.shuffle(local)
        selected.extend((int(image_idx) * MAX_CANDIDATES + int(j)) for j in local)
        coverage["background_selected"] += len(bg)
        coverage["ignored_not_selected"] += int(np.sum(groups < 0)) + max(0, int(np.sum(groups == 2)) - len(bg))
    if coverage["background_selected"] > coverage["foreground_selected"]:
        raise RuntimeError("background sampling cap exceeded 1:1")
    if not selected:
        raise RuntimeError("no eligible training candidates")
    return np.asarray(selected, dtype=np.int64), coverage, rng.bit_generator.state


def make_checkpoint(head, optimizer, epoch, global_step, next_offset, epoch_order,
                    sampler_state, epoch_metrics, active_gpu_seconds, accumulated_microbatches=0):
    if accumulated_microbatches != 0:
        raise RuntimeError("checkpoint requested away from a safe optimizer update boundary")
    config_path = ROOT / "RUN_CONFIG.json"
    return {
        "kind": "DMKPD-V31-corner-residual-head",
        "seed": SEED,
        "epoch": int(epoch),
        "epoch_complete": bool(next_offset >= len(epoch_order)),
        "global_step": int(global_step),
        "next_sample_offset": int(next_offset),
        "epoch_order": torch.from_numpy(epoch_order.astype(np.int64).copy()),
        "sampler_rng_state": sampler_state,
        "residual_head": to_cpu(head.state_dict()),
        "optimizer": to_cpu(optimizer.state_dict()),
        "fixed_scheduler": {"type": "none_fixed_lr", "learning_rate": LR},
        "rng": capture_rng(),
        "gradient_accumulation": {"microbatch": MICROBATCH, "steps": ACCUMULATION,
                                  "effective_max_batch": EFFECTIVE_BATCH,
                                  "pending_microbatches": 0, "safe_update_boundary": True},
        "epoch_metrics": epoch_metrics,
        "GPU_active_seconds_total": float(active_gpu_seconds),
        "detector_sha256": BASE_SHA,
        "quality_head_sha256": HEAD_SHA,
        "manifest_sha256": sha256(ROOT.parent / "07_第5版_R6正式复跑_20261006" / "inputs" / "split_manifest_r2_internal.csv"),
        "run_config_sha256": sha256(config_path),
    }


def save_checkpoint(head, optimizer, epoch, global_step, next_offset, epoch_order,
                    sampler_state, epoch_metrics, active_gpu_seconds, latest=True):
    bundle = make_checkpoint(head, optimizer, epoch, global_step, next_offset, epoch_order,
                             sampler_state, epoch_metrics, active_gpu_seconds)
    if latest:
        atomic_torch_save(bundle, ROOT / "checkpoints" / "latest.pt")
    atomic_torch_save(bundle, ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt")
    return bundle


def loss_terms(pred_norm: torch.Tensor, target_norm: torch.Tensor, groups: torch.Tensor):
    fg_mask = groups == 1
    bg_mask = groups == 2
    zero = pred_norm.sum() * 0.0
    fg_loss = F.smooth_l1_loss(pred_norm[fg_mask], target_norm[fg_mask], reduction="none").mean() if fg_mask.any() else zero
    bg_loss = F.smooth_l1_loss(pred_norm[bg_mask], torch.zeros_like(pred_norm[bg_mask]), reduction="none").mean() if bg_mask.any() else zero
    total = fg_loss + BACKGROUND_WEIGHT * bg_loss
    return total, fg_loss, bg_loss, int(fg_mask.sum()), int(bg_mask.sum())


def train(resume=True):
    if sha256(BASE) != BASE_SHA or sha256(Path(r"<PROJECT_ROOT>\DMKPD-V10-20261010\train\geometry_seed20261010\epoch_10.pt")) != HEAD_SHA:
        raise RuntimeError("frozen model hash mismatch before training")
    checks_path = ROOT / "diagnostics" / "IMPLEMENTATION_CHECKS.json"
    if not checks_path.exists() or not json.loads(checks_path.read_text(encoding="utf-8")).get("all_pass"):
        raise RuntimeError("required implementation checks are absent or failed")
    features, maps = load_train_maps()
    config_sha = sha256(ROOT / "RUN_CONFIG.json")
    set_v28_backend()
    seed_all(SEED)
    head = CornerResidualHead().cuda().train()
    trainable = [{"name": n, "shape": list(p.shape), "numel": p.numel()} for n, p in head.named_parameters() if p.requires_grad]
    if sum(x["numel"] for x in trainable) >= 1_000_000:
        raise RuntimeError("new residual branch exceeds 1M trainable parameter limit")
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    global_step = 0
    active_gpu_seconds = 0.0
    start_epoch = 1
    cursor = 0
    epoch_order = np.empty(0, dtype=np.int64)
    sampler_state = None
    epoch_metrics = {}
    latest_path = ROOT / "checkpoints" / "latest.pt"
    if resume and latest_path.exists():
        state = safe_torch_load(latest_path, "cpu")
        if state.get("kind") != "DMKPD-V31-corner-residual-head" or state.get("run_config_sha256") != config_sha:
            raise RuntimeError("resume checkpoint is not for this frozen V31 configuration")
        if state.get("detector_sha256") != BASE_SHA or state.get("quality_head_sha256") != HEAD_SHA:
            raise RuntimeError("resume checkpoint frozen input hash mismatch")
        head.load_state_dict(state["residual_head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        active_gpu_seconds = float(state["GPU_active_seconds_total"])
        epoch_metrics = state.get("epoch_metrics", {})
        if state["epoch_complete"]:
            start_epoch = int(state["epoch"]) + 1
        else:
            start_epoch = int(state["epoch"])
            epoch_order = state["epoch_order"].numpy().astype(np.int64)
            cursor = int(state["next_sample_offset"])
            sampler_state = state.get("sampler_rng_state")
        restore_rng(state["rng"])
        if state["epoch_complete"]:
            evaluation_status = ROOT / "results" / "epochs" / f"epoch_{int(state['epoch']):02d}_evaluation_status.json"
            if not evaluation_status.exists():
                # A process may stop after the epoch checkpoint is committed
                # but before its fixed evaluation finishes. Resume that
                # evaluation before moving to the next epoch; never repeat an
                # optimizer update or silently skip an epoch result.
                from evaluate_v31 import evaluate_epoch
                evaluate_epoch(int(state["epoch"]), latest_path,
                               do_bootstrap=(int(state["epoch"]) == EPOCHS))
    append_jsonl(ROOT / "logs" / "execution.jsonl", {"stage": "formal_training_start", "resume": resume,
                                                        "start_epoch": start_epoch, "global_step": global_step,
                                                        "trainable_parameters": trainable,
                                                        "parameter_count": sum(x["numel"] for x in trainable),
                                                        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
                                                        "microbatch": MICROBATCH, "accumulation": ACCUMULATION,
                                                        "effective_batch_max": EFFECTIVE_BATCH})
    for epoch in range(start_epoch, EPOCHS + 1):
        if not len(epoch_order) or (epoch != start_epoch) or cursor == 0:
            epoch_order, coverage, sampler_state = make_epoch_order(maps, epoch)
            cursor = 0
            order_path = ROOT / "cache" / "train_fit" / f"epoch_{epoch:02d}_sample_order.npy"
            np.save(order_path, epoch_order)
            dump_json(ROOT / "cache" / "train_fit" / f"epoch_{epoch:02d}_sampling.json",
                      {"epoch": epoch, "seed": SEED + 10007 * epoch, "coverage": coverage,
                       "sample_order_sha256": sha256(order_path), "background_cap_per_image": "min(background_pool, foreground_count)"})
        else:
            coverage = json.loads((ROOT / "cache" / "train_fit" / f"epoch_{epoch:02d}_sampling.json").read_text(encoding="utf-8"))["coverage"]
        epoch_start_time = time.perf_counter()
        epoch_gpu_s = 0.0
        sums = {"loss": 0.0, "fg": 0.0, "bg": 0.0, "fg_batches": 0, "bg_batches": 0,
                "selected_fg": 0, "selected_bg": 0, "updates": 0, "out_saturated": 0,
                "out_values": 0, "background_drift_norm_sum": 0.0, "background_drift_count": 0}
        grad_summary = {n: {"sum": 0.0, "count": 0, "max": 0.0} for n in ("fc1", "fc2", "out")}
        epoch_log = ROOT / "logs" / f"train_epoch_{epoch:02d}.jsonl"
        while cursor < len(epoch_order):
            effective_rows = epoch_order[cursor:cursor + EFFECTIVE_BATCH]
            chunk_outputs = []
            chunk_targets = []
            chunk_groups = []
            optimizer.zero_grad(set_to_none=True)
            ev_start = torch.cuda.Event(enable_timing=True); ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            for micro_start in range(0, len(effective_rows), MICROBATCH):
                micro_rows = effective_rows[micro_start:micro_start + MICROBATCH]
                # epoch_order stores flattened image×candidate rows. The
                # feature memmap is indexed as [image, candidate, C, H, W].
                image_rows, candidate_slots = decode_flat_candidate_rows(micro_rows, features.shape[0])
                feats = torch.from_numpy(np.asarray(features[image_rows, candidate_slots], dtype=np.float32).copy()).cuda(non_blocking=False)
                corner_norm = torch.from_numpy(np.asarray(maps["corner_norm"].reshape(-1, 4, 2)[micro_rows], dtype=np.float32)).cuda()
                target = torch.from_numpy(np.asarray(maps["target_norm"].reshape(-1, 4, 2)[micro_rows], dtype=np.float32)).cuda()
                groups = torch.from_numpy(np.asarray(maps["group"].reshape(-1)[micro_rows], dtype=np.int64)).cuda()
                if not torch.isfinite(feats).all() or not torch.isfinite(corner_norm).all() or not torch.isfinite(target).all():
                    raise FloatingPointError(f"nonfinite cached training data at epoch {epoch} offset {cursor}")
                z = head(feats, corner_norm).reshape(-1, 4, 2)
                chunk_outputs.append(0.05 * torch.tanh(z))
                chunk_targets.append(target)
                chunk_groups.append(groups)
            pred_norm = torch.cat(chunk_outputs, dim=0)
            targets_norm = torch.cat(chunk_targets, dim=0)
            group_tensor = torch.cat(chunk_groups, dim=0)
            loss, fg_loss, bg_loss, fg_count, bg_count = loss_terms(pred_norm, targets_norm, group_tensor)
            if not torch.isfinite(loss):
                torch.cuda.synchronize()
                save_checkpoint(head, optimizer, epoch, global_step, cursor, epoch_order,
                                sampler_state, epoch_metrics, active_gpu_seconds)
                dump_json(ROOT / "logs" / "NONFINITE_LOSS.json", {"epoch": epoch, "cursor": cursor,
                                                                   "fg_loss": float(fg_loss.detach().cpu()),
                                                                   "bg_loss": float(bg_loss.detach().cpu()),
                                                                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
                raise FloatingPointError("nonfinite V31 residual loss")
            loss.backward()
            layer_grad = {}
            for layer_name, layer in (("fc1", head.fc1), ("fc2", head.fc2), ("out", head.out)):
                grad_norm = math.sqrt(sum(float(p.grad.detach().float().pow(2).sum().cpu()) for p in layer.parameters() if p.grad is not None))
                layer_grad[layer_name] = grad_norm
                grad_summary[layer_name]["sum"] += grad_norm
                grad_summary[layer_name]["count"] += 1
                grad_summary[layer_name]["max"] = max(grad_summary[layer_name]["max"], grad_norm)
            output_weight_before = head.out.weight.detach().clone()
            output_bias_before = head.out.bias.detach().clone()
            optimizer.step()
            ev_end.record(); ev_end.synchronize()
            update_gpu_s = float(ev_start.elapsed_time(ev_end)) / 1000.0
            active_gpu_seconds += update_gpu_s
            epoch_gpu_s += update_gpu_s
            output_update = float((head.out.weight.detach() - output_weight_before).norm().cpu() +
                                  (head.out.bias.detach() - output_bias_before).norm().cpu())
            # Include feature transfers, forward, loss, backward, and optimizer
            # update in the active-GPU budget. The CUDA event spans work on the
            # default stream; host-only sampling/log formatting remains separate.
            fg_mask = group_tensor == 1
            bg_mask = group_tensor == 2
            sums["loss"] += float(loss.detach().cpu())
            sums["fg"] += float(fg_loss.detach().cpu()) if fg_count else 0.0
            sums["bg"] += float(bg_loss.detach().cpu()) if bg_count else 0.0
            sums["fg_batches"] += int(fg_count > 0); sums["bg_batches"] += int(bg_count > 0)
            sums["selected_fg"] += fg_count; sums["selected_bg"] += bg_count
            sums["updates"] += 1
            sums["out_saturated"] += int((pred_norm.detach().abs() >= 0.049).sum().cpu())
            sums["out_values"] += int(pred_norm.numel())
            if bg_mask.any():
                bg_drift = pred_norm.detach()[bg_mask]
                sums["background_drift_norm_sum"] += float(bg_drift.abs().mean().cpu())
                sums["background_drift_count"] += 1
            global_step += 1
            cursor += len(effective_rows)
            logrow = {"epoch": epoch, "global_step": global_step, "next_sample_offset": cursor,
                      "batch_rows": len(effective_rows), "fg_count": fg_count, "bg_count": bg_count,
                      "loss": float(loss.detach().cpu()), "loss_fg": float(fg_loss.detach().cpu()),
                      "loss_bg": float(bg_loss.detach().cpu()), "lr": LR, "weight_decay": WEIGHT_DECAY,
                      "gradient_norm_by_layer": layer_grad, "output_layer_update_norm": output_update,
                      "saturation_fraction_batch": float((pred_norm.detach().abs() >= 0.049).float().mean().cpu()),
                      "gpu_active_seconds": update_gpu_s, "accumulated_microbatches": len(chunk_outputs),
                      "safe_checkpoint_boundary": True}
            append_jsonl(epoch_log, logrow)
            if global_step % CHECKPOINT_EVERY_STEPS == 0:
                safe_torch_save = __import__("v31_core").atomic_torch_save
                bundle = make_checkpoint(head, optimizer, epoch, global_step, cursor, epoch_order,
                                         sampler_state, epoch_metrics, active_gpu_seconds)
                safe_torch_save(bundle, latest_path)
            if global_step % 100 == 0:
                print(json.dumps({"stage": "training", "epoch": epoch, "global_step": global_step,
                                  "cursor": cursor, "samples": len(epoch_order), "loss": logrow["loss"],
                                  "gpu_active_seconds_total": active_gpu_seconds}, ensure_ascii=False), flush=True)
        epoch_wall = time.perf_counter() - epoch_start_time
        avg_grads = {name: {"mean": vals["sum"] / max(vals["count"], 1), "max": vals["max"]} for name, vals in grad_summary.items()}
        epoch_metrics = {
            "epoch": epoch, "samples_selected": int(len(epoch_order)), "sampling_coverage": coverage,
            "selected_foreground": sums["selected_fg"], "selected_background": sums["selected_bg"],
            "optimizer_updates": sums["updates"], "loss_mean_per_update": sums["loss"] / max(sums["updates"], 1),
            "loss_fg_mean_per_update_with_fg": sums["fg"] / max(sums["fg_batches"], 1),
            "loss_bg_mean_per_update_with_bg": sums["bg"] / max(sums["bg_batches"], 1),
            "gradient_norm_by_layer": avg_grads,
            "output_saturation_fraction": sums["out_saturated"] / max(sums["out_values"], 1),
            "background_predicted_abs_norm_mean_per_batch": sums["background_drift_norm_sum"] / max(sums["background_drift_count"], 1),
            "lr": LR, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW",
            "microbatch": MICROBATCH, "accumulation_steps": ACCUMULATION,
            "last_group_actual_microbatches": int(min(ACCUMULATION, math.ceil((len(epoch_order) % EFFECTIVE_BATCH or EFFECTIVE_BATCH) / MICROBATCH))),
            "GPU_active_seconds_epoch": epoch_gpu_s, "GPU_active_seconds_total": active_gpu_seconds,
            "wall_seconds_epoch": epoch_wall,
        }
        bundle = save_checkpoint(head, optimizer, epoch, global_step, len(epoch_order), epoch_order,
                                 sampler_state, epoch_metrics, active_gpu_seconds)
        epoch_head = {"kind": "DMKPD-V31-residual-head", "epoch": epoch, "seed": SEED,
                      "residual_head": to_cpu(head.state_dict()), "model_config": {"channels": 256, "grid": GRID,
                          "hidden": 128, "hidden2": 64, "max_offset_roi_fraction": 0.05},
                      "detector_sha256": BASE_SHA, "quality_head_sha256": HEAD_SHA,
                      "manifest_sha256": bundle["manifest_sha256"], "run_config_sha256": config_sha}
        atomic_torch_save(epoch_head, ROOT / "checkpoints" / f"residual_head_epoch_{epoch:02d}.pt")
        dump_json(ROOT / "logs" / f"train_epoch_{epoch:02d}_summary.json", epoch_metrics)
        append_jsonl(ROOT / "logs" / "execution.jsonl", {"stage": "epoch_complete", **epoch_metrics})
        print(json.dumps({"stage": "epoch_complete", **epoch_metrics}, ensure_ascii=False), flush=True)
        # The frozen evaluation code is called in a fixed sequence after every
        # epoch. No training branch or optimizer logic consumes dev metrics.
        from evaluate_v31 import evaluate_epoch
        evaluate_epoch(epoch, ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt", do_bootstrap=(epoch == 3))
        epoch_order = np.empty(0, dtype=np.int64)
        cursor = 0
    final = safe_torch_load(ROOT / "checkpoints" / "latest.pt", "cpu")
    dump_json(ROOT / "logs" / "training_complete.json", {"status": "complete" if final.get("epoch") == EPOCHS and final.get("epoch_complete") else "incomplete",
                                                          "final_epoch": final.get("epoch"), "global_step": final.get("global_step"),
                                                          "GPU_active_seconds_total": final.get("GPU_active_seconds_total"),
                                                          "selected_model_epoch_fixed": EPOCHS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.resume and args.fresh:
        raise ValueError("choose only one of --resume or --fresh")
    train(resume=not args.fresh)


if __name__ == "__main__":
    main()
