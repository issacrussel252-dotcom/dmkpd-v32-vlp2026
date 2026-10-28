from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXP_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, choices=(20261031, 20261101))
parser.add_argument("--arm", choices=("A", "B"), required=True)
parser.add_argument("--resume", action="store_true")
_args = parser.parse_args()
RUN_ROOT = EXP_ROOT / "runs" / f"seed_{_args.seed}_{_args.arm}"
os.environ["DMKPD_V32_RUN_ROOT"] = str(RUN_ROOT)
os.environ["DMKPD_V32_SOURCE_ROOT"] = r"<PROJECT_ROOT>\DMKPD-V31-V28学习式角点残差"
sys.path.insert(0, str(EXP_ROOT / "code"))
sys.path.insert(0, str(EXP_ROOT / "code" / "source_snapshot"))
sys.path.insert(0, str(EXP_ROOT / "code" / "source_refs"))

from v31_core import (BASE, BASE_SHA, HEAD_CKPT, HEAD_SHA, MAX_CANDIDATES, ROOT, CornerResidualHead,
                      append_jsonl, atomic_torch_save, capture_rng, decode_flat_candidate_rows, dump_json,
                      restore_rng, safe_torch_load, seed_all, set_v28_backend, sha256)

SOURCE = Path(os.environ["DMKPD_V32_SOURCE_ROOT"])
EPOCHS = 3
MICROBATCH = 256
ACCUMULATION = 2
EFFECTIVE_BATCH = MICROBATCH * ACCUMULATION
LR = 1e-4
WEIGHT_DECAY = 1e-4
BG_WEIGHT = 0.1
CHECKPOINT_EVERY = 100


def to_cpu(obj):
    if isinstance(obj, torch.Tensor): return obj.detach().cpu()
    if isinstance(obj, dict): return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple): return tuple(to_cpu(v) for v in obj)
    return obj


def load_maps():
    cache = SOURCE / "cache" / "train_fit"
    names = ("group", "target_norm", "corner_norm", "candidate_count", "matched_small", "matched_tiny")
    return np.load(cache / "features.npy", mmap_mode="r"), {n: np.load(cache / f"{n}.npy", mmap_mode="r") for n in names}


def loss_terms(pred, target, groups, arm, alpha):
    fg = groups == 1
    bg = groups == 2
    zero = pred.sum() * 0.0
    beta = 1.0 if arm == "A" else 0.01
    fg_loss = F.smooth_l1_loss(pred[fg], target[fg], reduction="none", beta=beta).mean() if bool(fg.any()) else zero
    bg_loss = F.smooth_l1_loss(pred[bg], torch.zeros_like(pred[bg]), reduction="none", beta=1.0).mean() if bool(bg.any()) else zero
    total = (1.0 if arm == "A" else alpha) * fg_loss + BG_WEIGHT * bg_loss
    return total, fg_loss, bg_loss, int(fg.sum()), int(bg.sum())


def batch_tensors(flat_rows, features, maps):
    image_rows, slots = decode_flat_candidate_rows(flat_rows, features.shape[0])
    f = torch.from_numpy(np.asarray(features[image_rows, slots], dtype=np.float32).copy()).cuda()
    coords = torch.from_numpy(np.asarray(maps["corner_norm"].reshape(-1, 4, 2)[flat_rows], dtype=np.float32)).cuda()
    target = torch.from_numpy(np.asarray(maps["target_norm"].reshape(-1, 4, 2)[flat_rows], dtype=np.float32)).cuda()
    groups = torch.from_numpy(np.asarray(maps["group"].reshape(-1)[flat_rows], dtype=np.int64)).cuda()
    return f, coords, target, groups


def checkpoint(head, optimizer, epoch, step, cursor, order, sampler_state, sums, grad_summary,
               gpu_total, epoch_gpu, epoch_wall_start):
    if sums.get("pending_microbatches", 0) != 0:
        raise RuntimeError("checkpoint outside optimizer update boundary")
    cfg = ROOT / "RUN_CONFIG.json"
    return {"kind": "DMKPD-V32-corner-residual-head", "seed": _args.seed, "arm": _args.arm,
        "epoch": int(epoch), "epoch_complete": bool(cursor >= len(order)), "global_step": int(step),
        "next_sample_offset": int(cursor), "epoch_order": torch.from_numpy(order.astype(np.int64).copy()),
        "sampler_rng_state": sampler_state, "residual_head": to_cpu(head.state_dict()),
        "optimizer": to_cpu(optimizer.state_dict()), "fixed_scheduler": {"type": "none_fixed_lr", "learning_rate": LR},
        "rng": capture_rng(), "gradient_accumulation": {"microbatch": MICROBATCH, "steps": ACCUMULATION,
            "effective_max_batch": EFFECTIVE_BATCH, "pending_microbatches": 0, "safe_update_boundary": True},
        "epoch_sums": sums, "epoch_gradient_summary": grad_summary,
        "GPU_active_seconds_total": float(gpu_total), "GPU_active_seconds_epoch": float(epoch_gpu),
        "epoch_wall_start_monotonic_unusable_after_restart": float(epoch_wall_start),
        "detector_sha256": BASE_SHA, "quality_head_sha256": HEAD_SHA,
        "run_config_sha256": sha256(cfg), "sample_order_sha256": sha256(EXP_ROOT / "sampling_orders" / f"seed_{_args.seed}" / f"epoch_{epoch:02d}.npy"),
        "initialization_sha256": json.loads((EXP_ROOT / "INITIALIZATION_HASHES.json").read_text(encoding="utf-8"))[str(_args.seed)]["sha256"],
        "loss_definition": "A: fg SmoothL1 beta=1; B: alpha*fg SmoothL1 beta=.01; both + .1*bg SmoothL1 beta=1"}


def save_state(head, optimizer, epoch, step, cursor, order, sampler_state, sums, grad_summary,
               gpu_total, epoch_gpu, epoch_wall_start, epoch_boundary=False):
    bundle = checkpoint(head, optimizer, epoch, step, cursor, order, sampler_state, sums, grad_summary,
                        gpu_total, epoch_gpu, epoch_wall_start)
    atomic_torch_save(bundle, ROOT / "checkpoints" / "latest.pt")
    if epoch_boundary:
        atomic_torch_save(bundle, ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt")
        head_payload = {"kind": "DMKPD-V32-residual-head", "seed": _args.seed, "arm": _args.arm,
                        "epoch": epoch, "residual_head": to_cpu(head.state_dict()), "detector_sha256": BASE_SHA,
                        "quality_head_sha256": HEAD_SHA, "run_config_sha256": bundle["run_config_sha256"]}
        atomic_torch_save(head_payload, ROOT / "checkpoints" / f"residual_head_epoch_{epoch:02d}.pt")
    return bundle


def init_diagnostics(head, features, maps, alpha, epoch_tag="init"):
    diag_ids = np.load(EXP_ROOT / "diagnostics" / "fixed_train_fit_diagnostic_rows.npy")
    f, coords, target, groups = batch_tensors(diag_ids, features, maps)
    out = {}
    for label, mask in (("foreground_only", groups == 1), ("background_only", groups == 2)):
        head.zero_grad(set_to_none=True)
        pred = 0.05 * torch.tanh(head(f, coords).reshape(-1, 4, 2))
        beta = 1.0 if _args.arm == "A" or label == "background_only" else 0.01
        target_for_loss = target[mask] if label == "foreground_only" else torch.zeros_like(pred[mask])
        loss = F.smooth_l1_loss(pred[mask], target_for_loss, reduction="none", beta=beta).mean()
        if label == "foreground_only" and _args.arm == "B": loss = alpha * loss
        output_grad = torch.autograd.grad(loss, pred, retain_graph=True)[0]
        grads = torch.autograd.grad(loss, tuple(head.parameters()), allow_unused=False)
        by_layer = {}
        for name, module in (("fc1", head.fc1), ("fc2", head.fc2), ("out", head.out)):
            params = list(module.parameters())
            vals = [g for p, g in zip(head.parameters(), grads) if any(p is q for q in params)]
            by_layer[name] = float(torch.sqrt(sum(g.detach().float().pow(2).sum() for g in vals)).cpu())
        out[label] = {"loss": float(loss.detach().cpu()), "output_gradient_rms": float(torch.sqrt(output_grad.detach().float().pow(2).mean()).cpu()),
                      "parameter_grad_l2_by_layer": by_layer, "sample_rows": int(mask.sum())}
    with torch.no_grad():
        pred = 0.05 * torch.tanh(head(f, coords).reshape(-1, 4, 2))
        small_map = torch.from_numpy(np.asarray(maps["matched_small"].reshape(-1)[diag_ids], dtype=np.uint8)).cuda().bool()
        tiny_map = torch.from_numpy(np.asarray(maps["matched_tiny"].reshape(-1)[diag_ids], dtype=np.uint8)).cuda().bool()
        groups_out = groups == 1
        err = (pred-target).abs(); atarget = target.abs()
        for name, candidate_mask in (("all_foreground", groups_out), ("small_foreground", groups_out & small_map),
                                     ("tiny_foreground", groups_out & tiny_map)):
            component_mask = candidate_mask[:,None,None].expand_as(err)
            large_mask = component_mask & (atarget > .05)
            within_mask = component_mask & (atarget <= .05)
            out[name] = {"candidate_count": int(candidate_mask.sum()),
                "target_abs_median": float(atarget[component_mask].median().cpu()) if component_mask.any() else None,
                "prediction_abs_mean": float(pred.abs()[component_mask].mean().cpu()) if component_mask.any() else None,
                "error_abs_mean": float(err[component_mask].mean().cpu()) if component_mask.any() else None,
                "error_abs_mean_target_gt_005": float(err[large_mask].mean().cpu()) if large_mask.any() else None,
                "error_abs_mean_target_le_005": float(err[within_mask].mean().cpu()) if within_mask.any() else None,
                "output_saturation_fraction": float((pred.abs()[component_mask] >= .049).float().mean().cpu()) if component_mask.any() else None}
    head.zero_grad(set_to_none=True)
    dump_json(ROOT / "diagnostics" / f"train_fit_gradient_probe_epoch_{epoch_tag}.json", out)


def train():
    if sha256(BASE) != BASE_SHA or sha256(HEAD_CKPT) != HEAD_SHA:
        raise RuntimeError("frozen detector / quality head hash mismatch")
    checks = json.loads((EXP_ROOT / "diagnostics" / "IMPLEMENTATION_CHECKS.json").read_text(encoding="utf-8"))
    if not checks.get("all_pass"):
        raise RuntimeError("V32 implementation checks absent or failed")
    if not (ROOT / "RUN_CONFIG.json").exists() or sha256(ROOT / "RUN_CONFIG.json") != sha256(EXP_ROOT / "RUN_CONFIG.json"):
        raise RuntimeError("per-run frozen config is missing or differs")
    alpha = float(json.loads((EXP_ROOT / "ALPHA_CALIBRATION.json").read_text(encoding="utf-8"))["alpha"])
    init_path = EXP_ROOT / "initialization" / f"seed_{_args.seed}_initial.pt"
    init = safe_torch_load(init_path, "cpu")
    features, maps = load_maps()
    set_v28_backend()
    head = CornerResidualHead().cuda().train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    global_step = 0
    active_gpu_total = 0.0
    epoch_order = np.empty(0, dtype=np.int64)
    sampler_state = None
    cursor = 0
    start_epoch = 1
    sums = {}
    grad_summary = {}
    epoch_gpu = 0.0
    epoch_wall_start = time.perf_counter()
    latest = ROOT / "checkpoints" / "latest.pt"
    if latest.exists() and not _args.resume:
        raise RuntimeError(f"existing checkpoint requires explicit --resume: {latest}")
    if latest.exists():
        state = safe_torch_load(latest, "cpu")
        if state.get("run_config_sha256") != sha256(ROOT / "RUN_CONFIG.json") or state.get("seed") != _args.seed or state.get("arm") != _args.arm:
            raise RuntimeError("checkpoint does not match this arm/config")
        head.load_state_dict(state["residual_head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state["global_step"])
        active_gpu_total = float(state["GPU_active_seconds_total"])
        restore_rng(state["rng"])
        if state["epoch_complete"]:
            start_epoch = int(state["epoch"]) + 1
        else:
            start_epoch = int(state["epoch"])
            epoch_order = state["epoch_order"].numpy().astype(np.int64)
            cursor = int(state["next_sample_offset"])
            sampler_state = state["sampler_rng_state"]
            sums = state["epoch_sums"]
            grad_summary = state["epoch_gradient_summary"]
            epoch_gpu = float(state["GPU_active_seconds_epoch"])
        log_path = ROOT / "logs" / "training_updates.jsonl"
        if log_path.exists():
            kept = []
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if int(row.get("global_step", 0)) <= global_step: kept.append(line)
                except Exception: pass
            log_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        if state["epoch_complete"]:
            status = ROOT / "results" / "epochs" / f"epoch_{int(state['epoch']):02d}_evaluation_status.json"
            if not status.exists():
                from evaluate_v31 import evaluate_epoch
                evaluate_epoch(int(state["epoch"]), latest, do_bootstrap=False, do_online_if_pass=False)
    else:
        head.load_state_dict(init["state_dict"], strict=True)
        restore_rng(init["post_init_rng"])
        init_diagnostics(head, features, maps, alpha)
    if start_epoch > EPOCHS:
        dump_json(ROOT / "logs" / "training_complete.json", {"status": "complete", "final_epoch": EPOCHS,
            "global_step": global_step, "GPU_active_seconds_total": active_gpu_total})
        return

    append_jsonl(ROOT / "logs" / "execution.jsonl", {"stage": "training_start_or_resume", "seed": _args.seed,
        "arm": _args.arm, "start_epoch": start_epoch, "global_step": global_step, "alpha": alpha,
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "microbatch": MICROBATCH,
        "accumulation": ACCUMULATION, "resume": latest.exists()})
    for epoch in range(start_epoch, EPOCHS + 1):
        order_path = EXP_ROOT / "sampling_orders" / f"seed_{_args.seed}" / f"epoch_{epoch:02d}.npy"
        if epoch != start_epoch or len(epoch_order) == 0:
            epoch_order = np.load(order_path, mmap_mode="r").copy()
            sampler_record = json.loads(order_path.with_suffix(".json").read_text(encoding="utf-8"))
            sampler_state = sampler_record["sampler_rng_state"]
            cursor = 0
            epoch_gpu = 0.0
            epoch_wall_start = time.perf_counter()
            sums = {"loss_total_sum": 0.0, "loss_fg_sum": 0.0, "loss_bg_sum": 0.0, "updates": 0,
                    "fg_rows": 0, "bg_rows": 0, "coord_values": 0, "saturated_values": 0,
                    "large_target_error_before_sum": 0.0, "large_target_error_after_sum": 0.0,
                    "large_target_values": 0, "inrange_error_before_sum": 0.0, "inrange_error_after_sum": 0.0,
                    "inrange_values": 0, "background_abs_output_sum": 0.0, "background_values": 0,
                    "pending_microbatches": 0}
            grad_summary = {n: {"sum": 0.0, "max": 0.0, "count": 0} for n in ("fc1", "fc2", "out")}
        epoch_log = ROOT / "logs" / "training_updates.jsonl"
        while cursor < len(epoch_order):
            rows = np.asarray(epoch_order[cursor:cursor + EFFECTIVE_BATCH], dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True); ev0.record()
            preds, targets, groups_list = [], [], []
            for ms in range(0, len(rows), MICROBATCH):
                f, coords, target, groups = batch_tensors(rows[ms:ms + MICROBATCH], features, maps)
                if not torch.isfinite(f).all() or not torch.isfinite(target).all():
                    raise FloatingPointError(f"nonfinite train_fit inputs at seed={_args.seed} arm={_args.arm} epoch={epoch} offset={cursor}")
                z = head(f, coords).reshape(-1, 4, 2)
                preds.append(0.05 * torch.tanh(z)); targets.append(target); groups_list.append(groups)
            pred = torch.cat(preds); target = torch.cat(targets); groups = torch.cat(groups_list)
            loss, fg_loss, bg_loss, fg_count, bg_count = loss_terms(pred, target, groups, _args.arm, alpha)
            if not torch.isfinite(loss):
                torch.cuda.synchronize()
                raise FloatingPointError(f"nonfinite loss: seed={_args.seed}, arm={_args.arm}, epoch={epoch}, cursor={cursor}")
            loss.backward()
            layer_norms = {}
            for name, module in (("fc1", head.fc1), ("fc2", head.fc2), ("out", head.out)):
                norm = math.sqrt(sum(float(p.grad.detach().float().pow(2).sum().cpu()) for p in module.parameters() if p.grad is not None))
                layer_norms[name] = norm
                grad_summary[name]["sum"] += norm; grad_summary[name]["max"] = max(grad_summary[name]["max"], norm); grad_summary[name]["count"] += 1
            before_w = head.out.weight.detach().clone(); before_b = head.out.bias.detach().clone()
            optimizer.step()
            ev1.record(); ev1.synchronize()
            gpu_s = float(ev0.elapsed_time(ev1)) / 1000.0
            active_gpu_total += gpu_s; epoch_gpu += gpu_s
            update_norm = float((head.out.weight.detach() - before_w).norm().cpu() + (head.out.bias.detach() - before_b).norm().cpu())
            fgmask = groups == 1; bgmask = groups == 2
            abs_err = (pred.detach() - target).abs()
            abs_target = target.abs()
            fg_broadcast = fgmask[:, None, None].expand_as(abs_target)
            large = fg_broadcast & (abs_target > 0.05)
            inrange = fg_broadcast & (abs_target <= 0.05)
            if large.any():
                sums["large_target_error_before_sum"] += float(abs_target[large].sum().cpu())
                sums["large_target_error_after_sum"] += float(abs_err[large].sum().cpu())
                sums["large_target_values"] += int(large.sum())
            if inrange.any():
                sums["inrange_error_before_sum"] += float(abs_target[inrange].sum().cpu())
                sums["inrange_error_after_sum"] += float(abs_err[inrange].sum().cpu())
                sums["inrange_values"] += int(inrange.sum())
            if bgmask.any():
                sums["background_abs_output_sum"] += float(pred.detach()[bgmask].abs().sum().cpu())
                sums["background_values"] += int(pred[bgmask].numel())
            sums["loss_total_sum"] += float(loss.detach().cpu()); sums["loss_fg_sum"] += float(fg_loss.detach().cpu()); sums["loss_bg_sum"] += float(bg_loss.detach().cpu())
            sums["updates"] += 1; sums["fg_rows"] += fg_count; sums["bg_rows"] += bg_count
            sums["coord_values"] += int(pred.numel()); sums["saturated_values"] += int((pred.detach().abs() >= .049).sum().cpu())
            global_step += 1; cursor += len(rows)
            log = {"seed": _args.seed, "arm": _args.arm, "epoch": epoch, "global_step": global_step,
                "next_sample_offset": cursor, "batch_rows": int(len(rows)), "fg_rows": fg_count, "bg_rows": bg_count,
                "loss_total": float(loss.detach().cpu()), "loss_fg_unscaled": float(fg_loss.detach().cpu()),
                "loss_bg_unscaled": float(bg_loss.detach().cpu()), "fg_beta": 1.0 if _args.arm == "A" else .01,
                "fg_alpha": 1.0 if _args.arm == "A" else alpha, "bg_weight": BG_WEIGHT,
                "gradient_norm_by_layer": layer_norms, "output_layer_update_norm": update_norm,
                "output_saturation_fraction": float((pred.detach().abs() >= .049).float().mean().cpu()),
                "background_abs_output_mean": float(pred.detach()[bgmask].abs().mean().cpu()) if bgmask.any() else 0.0,
                "gpu_active_seconds": gpu_s, "microbatches": len(preds), "safe_update_boundary": True}
            append_jsonl(epoch_log, log)
            if global_step % CHECKPOINT_EVERY == 0:
                save_state(head, optimizer, epoch, global_step, cursor, epoch_order, sampler_state, sums, grad_summary,
                           active_gpu_total, epoch_gpu, epoch_wall_start)
            if global_step % 100 == 0:
                print(json.dumps({"stage": "training", "seed": _args.seed, "arm": _args.arm, "epoch": epoch,
                    "global_step": global_step, "cursor": cursor, "rows": len(epoch_order), "loss": log["loss_total"],
                    "gpu_active_seconds_total": active_gpu_total}, ensure_ascii=False), flush=True)
        summary = {"seed": _args.seed, "arm": _args.arm, "epoch": epoch, "sample_order_sha256": sha256(order_path),
            "selected_rows": int(len(epoch_order)), "optimizer_updates": int(sums["updates"]),
            "mean_update_total_loss": sums["loss_total_sum"] / max(sums["updates"], 1),
            "mean_update_foreground_loss_unscaled": sums["loss_fg_sum"] / max(sums["updates"], 1),
            "mean_update_background_loss_unscaled": sums["loss_bg_sum"] / max(sums["updates"], 1),
            "foreground_rows": sums["fg_rows"], "background_rows": sums["bg_rows"],
            "output_saturation_fraction": sums["saturated_values"] / max(sums["coord_values"], 1),
            "mean_abs_error_target_gt_005_before": sums["large_target_error_before_sum"] / max(sums["large_target_values"], 1),
            "mean_abs_error_target_gt_005_after": sums["large_target_error_after_sum"] / max(sums["large_target_values"], 1),
            "mean_abs_error_target_le_005_before": sums["inrange_error_before_sum"] / max(sums["inrange_values"], 1),
            "mean_abs_error_target_le_005_after": sums["inrange_error_after_sum"] / max(sums["inrange_values"], 1),
            "mean_background_abs_output": sums["background_abs_output_sum"] / max(sums["background_values"], 1),
            "gradient_norm_layer_mean_and_max": {n: {"mean": v["sum"] / max(v["count"], 1), "max": v["max"]} for n, v in grad_summary.items()},
            "lr": LR, "weight_decay": WEIGHT_DECAY, "optimizer": "AdamW", "microbatch": MICROBATCH,
            "accumulation": ACCUMULATION, "last_effective_batch_rows": int(len(epoch_order) % EFFECTIVE_BATCH or EFFECTIVE_BATCH),
            "GPU_active_seconds_epoch": epoch_gpu, "GPU_active_seconds_total": active_gpu_total,
            "wall_seconds_epoch": time.perf_counter() - epoch_wall_start, "loss_definition": "A beta=1; B alpha*beta=.01; background beta=1 weight=.1"}
        dump_json(ROOT / "logs" / f"epoch_{epoch:02d}_training_summary.json", summary)
        append_jsonl(ROOT / "logs" / "execution.jsonl", {"stage": "epoch_complete", **summary})
        init_diagnostics(head, features, maps, alpha, epoch_tag=f"{epoch:02d}")
        bundle = save_state(head, optimizer, epoch, global_step, len(epoch_order), epoch_order, sampler_state,
                            sums, grad_summary, active_gpu_total, epoch_gpu, epoch_wall_start, epoch_boundary=True)
        if epoch == 3:
            dump_json(ROOT / "diagnostics" / "final_state.json", {"seed": _args.seed, "arm": _args.arm,
                "checkpoint": str((ROOT / "checkpoints" / "epoch_03.pt").resolve()),
                "checkpoint_sha256": sha256(ROOT / "checkpoints" / "epoch_03.pt"),
                "residual_head_sha256": sha256(ROOT / "checkpoints" / "residual_head_epoch_03.pt"),
                "GPU_active_seconds_total": active_gpu_total})
        from evaluate_v31 import evaluate_epoch
        evaluate_epoch(epoch, ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt", do_bootstrap=False, do_online_if_pass=False)
        epoch_order = np.empty(0, dtype=np.int64); cursor = 0
    dump_json(ROOT / "logs" / "training_complete.json", {"status": "complete", "final_epoch": EPOCHS,
        "global_step": global_step, "GPU_active_seconds_total": active_gpu_total,
        "selected_model_epoch_fixed": 3, "evaluation_epoch3_status": str(ROOT / "results" / "epochs" / "epoch_03_evaluation_status.json")})


if __name__ == "__main__":
    train()
