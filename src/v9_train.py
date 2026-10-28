from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import os
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from v9_geometry import canonical_quad, ordered_quad_is_valid, parse_shapes, polygon_area, quad_iou
from v9_model import V9_KIND_A, V9_KIND_B, build_model, norm_inventory, parse_prediction

DEFAULT_MANIFEST = Path(os.environ.get('DMKPD_MANIFEST', 'data/manifest.csv'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def read_rows(manifest: Path, role: str) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("split") == role]
    return sorted(rows, key=lambda row: row["stem"].casefold())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Keep the last complete checkpoint intact if a process stops mid-write."""
    temporary = path.with_name(path.name + ".writing.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def prepare_initializations(args: argparse.Namespace) -> None:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    model_a = build_model(False, candidate_budget=args.candidate_budget, pretrained_backbone=True)
    rng_train = capture_rng()
    state_a = {k: v.detach().cpu().clone() for k, v in model_a.state_dict().items()}
    trainable_a = norm_inventory(model_a)
    del model_a

    seed_all(args.seed)
    model_b = build_model(True, candidate_budget=args.candidate_budget, pretrained_backbone=True)
    state_b = {k: v.detach().cpu().clone() for k, v in model_b.state_dict().items()}
    trainable_b = norm_inventory(model_b)
    shared = set(state_a).intersection(state_b)
    unequal = [k for k in sorted(shared) if not torch.equal(state_a[k], state_b[k])]
    if unequal:
        raise RuntimeError(f"paired initializations differ in common tensors: {unequal[:8]}")
    config = {
        "revision": "DMKPD-V9-20261009",
        "seed": args.seed,
        "candidate_budget_pre_keypoint": args.candidate_budget,
        "final_output_cap_after_polygon_nms": 100,
        "model": {
            "framework": "torchvision Keypoint R-CNN",
            "backbone": "ResNet50-FPN",
            "backbone_weights": "ResNet50_Weights.DEFAULT ImageNet",
            "normalization": "FrozenBatchNorm2d explicitly supplied to resnet_fpn_backbone for initialization, load, resume, train, and inference",
            "input_min_size": 768,
            "input_max_size": 1024,
            "box_score_threshold": 0.0,
            "box_nms_threshold": 1.0,
            "polygon_nms_iou": 0.30,
            "keypoints": 4,
            "quality_definition": "B only: conditional expected local OKS given foreground; positive sampled RoIs only; detached target measured on class-1 box-regression decoded/clipped boxes and actual Keypoint R-CNN predicted corners; evaluator canonicalization and local OKS; invalid/out-of-bounds predictions target 0; background quality loss and gradient 0",
            "primary_B_score": "p_cls*q",
            "score_ablation": ["p_cls", "q", "p_cls*q", "p_cls*sqrt(q)"],
            "pre_keypoint_pool_for_all_v9_score_formulas": "top candidate_budget RoIs by class probability after small-box filtering and box NMS; identical candidate IDs for all four formulas",
            "RoI_matcher": "standard torchvision Matcher with allow_low_quality_matches=False",
        },
        "train": {
            "role": "train_fit only",
            "epochs": 12,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "tail_accumulation": "divide by actual microbatch count in final partial group",
            "optimizer": "SGD(momentum=0.9)",
            "learning_rate": 0.0025,
            "weight_decay": 0.0001,
            "gradient_clip": 10.0,
            "scheduler": "CosineAnnealingLR(T_max=12, eta_min=0.00005)",
            "amp": "bfloat16 on CUDA for training; inference FP32",
            "augmentation": "seeded per (seed, epoch, image_index), horizontal flip p=0.5, brightness/contrast each p=0.35 and factor U[0.78,1.22]",
            "paired_seed": args.seed,
            "same_common_parameter_initialization": True,
            "same_detector_sampling_rng_start": True,
        },
        "inventory_A": trainable_a,
        "inventory_B": trainable_b,
        "common_parameter_tensor_count": len(shared),
        "common_parameter_sha256_A": state_digest(state_a),
        "common_parameter_sha256_B": state_digest({k: state_b[k] for k in shared}),
        "common_parameter_values_identical": True,
        "historical_detector_checkpoint_loaded": False,
    }
    torch.save({"checkpoint_kind": V9_KIND_A, "seed": args.seed, "model": state_a, "rng_state_for_training": rng_train, "config": config}, out / "init_A.pt")
    torch.save({"checkpoint_kind": V9_KIND_B, "seed": args.seed, "model": state_b, "rng_state_for_training": rng_train, "config": config}, out / "init_B.pt")
    write_json(out / "model_config.json", config)
    print(json.dumps({"status": "prepared", "output": str(out), "common_tensors": len(shared), "common_equal": not unequal, "norm_A": trainable_a["normalization_modules"], "norm_B": trainable_b["normalization_modules"], "sha256_A": state_digest(state_a), "sha256_B_common": state_digest({k: state_b[k] for k in shared})}, ensure_ascii=False), flush=True)


class TrainDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], seed: int, epoch: int = 0) -> None:
        self.rows, self.seed, self.epoch = rows, seed, epoch

    def __len__(self) -> int:
        return len(self.rows)

    def _aug(self, index: int) -> dict[str, float | bool]:
        rng = random.Random((self.seed << 32) ^ (self.epoch << 16) ^ index)
        return {"flip": rng.random() < 0.5, "brightness": rng.uniform(.78, 1.22) if rng.random() < .35 else 1.0, "contrast": rng.uniform(.78, 1.22) if rng.random() < .35 else 1.0}

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        orig_w, orig_h = image.size
        quads = parse_shapes(Path(row["annotation"]), row["stem"])
        augmentation = self._aug(index)
        if augmentation["flip"]:
            image = TF.hflip(image)
            quads = [canonical_quad([(orig_w - 1.0 - x, y) for x, y in quad]) for quad in quads]
        if augmentation["brightness"] != 1.0:
            image = ImageEnhance.Brightness(image).enhance(float(augmentation["brightness"]))
        if augmentation["contrast"] != 1.0:
            image = ImageEnhance.Contrast(image).enhance(float(augmentation["contrast"]))
        boxes, keypoints, areas, small, tiny = [], [], [], [], []
        for quad in quads:
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            box = [min(xs), min(ys), max(xs), max(ys)]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"non-positive box {row['stem']}")
            boxes.append(box)
            keypoints.append([[x, y, 2.0] for x, y in quad])
            areas.append((box[2] - box[0]) * (box[3] - box[1]))
            small.append(polygon_area(quad) / max(orig_w * orig_h, 1) < .01)
            tiny.append(min(math.dist(quad[i], quad[(i + 1) % 4]) for i in range(4)) < 16.0)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "keypoints": torch.tensor(keypoints, dtype=torch.float32),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "gt_small": torch.tensor(small, dtype=torch.bool),
            "gt_tiny": torch.tensor(tiny, dtype=torch.bool),
        }
        return TF.to_tensor(image), target, row["stem"]


def collate_train(batch):
    images, targets, stems = zip(*batch)
    return list(images), list(targets), list(stems)


def model_config_for_run(arm: str, args: argparse.Namespace, manifest: Path, rows: list[dict[str, str]], init_path: Path) -> dict[str, Any]:
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    return {
        "revision": "DMKPD-V9-20261009",
        "arm": arm,
        "checkpoint_kind": V9_KIND_A if arm == "A" else V9_KIND_B,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": 1,
        "gradient_accumulation": args.accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "optimizer": "SGD(momentum=0.9)",
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": args.epochs,
        "scheduler_eta_min": args.learning_rate * .02,
        "amp_bfloat16_cuda": bool(args.amp and str(args.device).startswith("cuda")),
        "candidate_budget_pre_keypoint": args.candidate_budget,
        "input_min_size": 768,
        "input_max_size": 1024,
        "polygon_nms_iou": .30,
        "roi_matcher_allow_low_quality_matches": False,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "train_role": "train_fit",
        "train_fit_images": len(rows),
        "train_annotation_sha256": {r["stem"]: sha256(Path(r["annotation"])) for r in rows},
        "initial_checkpoint": str(init_path),
        "initial_checkpoint_sha256": sha256(init_path),
        "initialization": "ImageNet backbone with new detection/keypoint heads; no V7/V8 or other detector checkpoint",
        "historical_detector_checkpoint_loaded": False,
        "initialization_norm_type": "FrozenBatchNorm2d",
        "norm_inventory": init["config"]["inventory_A" if arm == "A" else "inventory_B"],
        "same_shared_initialization": True,
        "same_global_train_rng_start": True,
        "augmentation": "epoch-indexed deterministic transform; horizontal flip p=.5; brightness and contrast p=.35, factor .78-1.22",
        "sampler": "uniform DataLoader shuffle, per epoch generator seed=seed+epoch",
        "quality_definition": None if arm == "A" else init["config"]["model"]["quality_definition"],
        "score_formula": "p_cls" if arm == "A" else "p_cls*q",
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = Path(args.manifest).resolve()
    rows = read_rows(manifest, "train_fit")
    if not rows:
        raise RuntimeError("train_fit has no rows")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    init_path = Path(args.init_checkpoint).resolve()
    init = torch.load(init_path, map_location="cpu", weights_only=False)
    arm = "A" if init["checkpoint_kind"] == V9_KIND_A else "B" if init["checkpoint_kind"] == V9_KIND_B else ""
    if not arm or int(init["seed"]) != args.seed:
        raise ValueError("initial state kind or seed mismatch")
    kind = init["checkpoint_kind"]
    run_cfg = output / "run_config.json"
    latest_path = output / "latest.pt"
    if args.resume:
        resume_path = Path(args.resume).resolve()
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume.get("checkpoint_kind") != kind or resume.get("seed") != args.seed:
            raise ValueError("resume checkpoint kind/seed mismatch")
        if resume.get("manifest_sha256") != sha256(manifest):
            raise ValueError("resume manifest hash mismatch")
        start_epoch = int(resume["epoch"])
    else:
        if latest_path.exists() or (output / "training_complete.json").exists():
            raise FileExistsError(f"output already initialized: {output}")
        start_epoch = 0
        write_json(run_cfg, model_config_for_run(arm, args, manifest, rows, init_path))

    model = build_model(arm == "B", candidate_budget=args.candidate_budget, pretrained_backbone=False).to(device)
    model.load_state_dict((resume["model"] if args.resume else init["model"]), strict=True)
    norm_info = norm_inventory(model)
    if norm_info["normalization_modules"] != {"FrozenBatchNorm2d": 53}:
        raise RuntimeError(f"expected exactly 53 FrozenBatchNorm2d modules, got {norm_info['normalization_modules']}")
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, momentum=.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * .02)
    if args.resume:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        restore_rng(resume["rng_state"])
        saved_config = json.loads(run_cfg.read_text(encoding="utf-8"))
        resume_config=resume.get("registered_config",{})
        expected={"manifest_sha256":sha256(manifest),"candidate_budget_pre_keypoint":args.candidate_budget,"epochs":args.epochs,"gradient_accumulation":args.accumulation,"learning_rate":args.learning_rate,"weight_decay":args.weight_decay,"gradient_clip":args.gradient_clip,"amp_bfloat16_cuda":bool(args.amp and device.type=="cuda"),"initial_checkpoint_sha256":sha256(init_path)}
        mismatch={k:(resume_config.get(k),saved_config.get(k),v) for k,v in expected.items() if saved_config.get(k)!=v or (k in resume_config and resume_config.get(k)!=v)}
        if mismatch:raise ValueError(f"resume training configuration mismatch: {mismatch}")
        if int(scheduler.last_epoch)!=start_epoch:raise ValueError(f"resume scheduler epoch {scheduler.last_epoch} != next epoch {start_epoch}")
        if int(resume["sampler_state"]["completed_epoch"])!=start_epoch or int(resume["sampler_state"]["next_epoch_seed"])!=args.seed+start_epoch:raise ValueError("resume sampler state is not at the next epoch boundary")
        if int(resume["augmentation_state"]["next_augmentation_epoch"])!=start_epoch:raise ValueError("resume augmentation epoch is not continuous")
    else:
        restore_rng(init["rng_state_for_training"])

    dataset = TrainDataset(rows, args.seed)
    log_path = output / "training_log.csv"
    quality_path = output / "quality_batch_stats.csv"
    log_fields = ["epoch", "images", "steps", "optimizer_updates", "loss", "loss_classifier", "loss_box_reg", "loss_keypoint", "loss_quality", "positive_rois", "negative_rois", "invalid_quality_targets", "positive_quality_gradient_mean", "negative_quality_gradient_mean", "lr", "seconds", "finite", "sampler_seed", "augmentation_epoch", "tail_group_size"]
    quality_fields = ["epoch", "step", "positive_rois", "negative_rois", "positive_loss", "negative_loss", "positive_q_mean", "positive_q_p10", "positive_q_p50", "positive_q_p90", "positive_target_mean", "positive_target_p10", "positive_target_p50", "positive_target_p90", "negative_q_mean", "negative_q_p10", "negative_q_p50", "negative_q_p90", "invalid_targets", "by_scale_json"]
    append = bool(args.resume)
    log_handle = log_path.open("a" if append else "w", encoding="utf-8-sig", newline="")
    q_handle = quality_path.open("a" if append else "w", encoding="utf-8-sig", newline="")
    writer, qwriter = csv.DictWriter(log_handle, fieldnames=log_fields), csv.DictWriter(q_handle, fieldnames=quality_fields)
    if not append:
        writer.writeheader(); qwriter.writeheader()
    try:
        for epoch in range(start_epoch, args.epochs):
            dataset.epoch = epoch
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.workers, collate_fn=collate_train, pin_memory=True, persistent_workers=False, generator=generator)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            sums: dict[str, float] = {}
            positives = negatives = invalid_targets = updates = 0
            pos_gradients: list[float] = []
            total_steps = len(loader)
            finite = True
            for step, (images, targets, _) in enumerate(loader, 1):
                images = [image.to(device, non_blocking=True) for image in images]
                targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
                group_start = ((step - 1) // args.accumulation) * args.accumulation
                actual_group = min(args.accumulation, total_steps - group_start)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp and device.type == "cuda")):
                    losses = model(images, targets)
                    raw_loss = sum(losses.values())
                    scaled_loss = raw_loss / actual_group
                if not torch.isfinite(raw_loss):
                    finite = False
                    raise FloatingPointError(f"non-finite loss at epoch={epoch+1} step={step}")
                scaled_loss.backward()
                stats = model.roi_heads.last_quality_stats
                if stats:
                    positives += int(stats["positive_rois"]); negatives += int(stats["negative_rois"])
                    invalid_targets += int(stats["invalid_or_out_of_bounds_positive_targets"])
                    def val(group, key):
                        v = stats[group].get(key)
                        return "" if v is None else v
                    qwriter.writerow({
                        "epoch": epoch+1, "step": step, "positive_rois": stats["positive_rois"], "negative_rois": stats["negative_rois"],
                        "positive_loss": stats["positive_quality_loss"], "negative_loss": stats["negative_quality_loss"],
                        "positive_q_mean": stats["positive_q_prediction"]["mean"], "positive_q_p10": stats["positive_q_prediction"]["p10"], "positive_q_p50": stats["positive_q_prediction"]["p50"], "positive_q_p90": stats["positive_q_prediction"]["p90"],
                        "positive_target_mean": stats["positive_q_target"]["mean"], "positive_target_p10": stats["positive_q_target"]["p10"], "positive_target_p50": stats["positive_q_target"]["p50"], "positive_target_p90": stats["positive_q_target"]["p90"],
                        "negative_q_mean": stats["negative_q_prediction"]["mean"], "negative_q_p10": stats["negative_q_prediction"]["p10"], "negative_q_p50": stats["negative_q_prediction"]["p50"], "negative_q_p90": stats["negative_q_prediction"]["p90"],
                        "invalid_targets": stats["invalid_or_out_of_bounds_positive_targets"], "by_scale_json": json.dumps(stats["by_scale"], separators=(",", ":")),
                    })
                should_step = step % args.accumulation == 0 or step == total_steps
                if should_step:
                    if arm == "B":
                        qnorm_sq = sum(float(p.grad.detach().float().norm().cpu()) ** 2 for p in model.roi_heads.quality_head.parameters() if p.grad is not None)
                        pos_gradients.append(math.sqrt(qnorm_sq))
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    if not torch.isfinite(grad_norm):
                        finite = False
                        raise FloatingPointError(f"non-finite gradient at epoch={epoch+1} step={step}")
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
                for name, value in losses.items():
                    sums[name] = sums.get(name, 0.0) + float(value.detach().float().cpu())
                if step == 1 or step % args.progress_interval == 0 or step == total_steps:
                    log_handle.flush(); q_handle.flush()
                    print(json.dumps({"arm": arm, "epoch": epoch+1, "step": step, "steps": total_steps, "loss": float(raw_loss.detach().cpu()), "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter()-start, "positive_rois": positives, "negative_rois": negatives, "augmentation_epoch": epoch}, ensure_ascii=False), flush=True)
            scheduler.step()
            elapsed = time.perf_counter() - start
            row = {
                "epoch": epoch+1, "images": len(dataset), "steps": total_steps, "optimizer_updates": updates,
                "loss": sum(sums.values()) / max(total_steps, 1),
                "loss_classifier": sums.get("loss_classifier", 0.0)/max(total_steps,1), "loss_box_reg": sums.get("loss_box_reg",0.0)/max(total_steps,1),
                "loss_keypoint": sums.get("loss_keypoint",0.0)/max(total_steps,1), "loss_quality": sums.get("loss_quality",0.0)/max(total_steps,1),
                "positive_rois": positives, "negative_rois": negatives, "invalid_quality_targets": invalid_targets,
                "positive_quality_gradient_mean": statistics.mean(pos_gradients) if pos_gradients else 0.0,
                "negative_quality_gradient_mean": 0.0, "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed,
                "finite": int(finite), "sampler_seed": args.seed+epoch, "augmentation_epoch": epoch,
                "tail_group_size": total_steps % args.accumulation or args.accumulation,
            }
            writer.writerow(row); log_handle.flush()
            # Per-epoch inference weights are self-contained. Optimizer/scheduler/
            # RNG and sampler state live in latest.pt for space-efficient resume.
            epoch_payload = {"checkpoint_kind": kind, "revision": "DMKPD-V9-20261009", "epoch": epoch+1, "seed": args.seed, "model": model.state_dict(), "model_config": run_cfg.read_text(encoding="utf-8")}
            epoch_path = output / f"epoch_{epoch+1:02d}.pt"
            atomic_torch_save(epoch_payload, epoch_path)
            latest_payload = {
                "checkpoint_kind": kind, "revision": "DMKPD-V9-20261009", "epoch": epoch+1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng_state": capture_rng(),
                "sampler_state": {"completed_epoch": epoch+1, "next_epoch_seed": args.seed+epoch+1, "epoch_generator_state": generator.get_state()},
                "augmentation_state": {"completed_epoch": epoch, "next_augmentation_epoch": epoch+1, "deterministic_key": "(seed << 32) ^ (epoch << 16) ^ image_index"},
                "seed": args.seed, "manifest_sha256": sha256(manifest), "train_images": len(dataset),
                "init_checkpoint_sha256": sha256(init_path), "registered_config": json.loads(run_cfg.read_text(encoding="utf-8")),
            }
            atomic_torch_save(latest_payload, latest_path)
            print(json.dumps({**row, "checkpoint": str(epoch_path), "latest": str(latest_path)}, ensure_ascii=False), flush=True)
            del loader
        write_json(output / "training_complete.json", {"status": "complete", "arm": arm, "epochs": args.epochs, "seed": args.seed, "last_checkpoint": str(output / f"epoch_{args.epochs:02d}.pt"), "resumable_state": str(latest_path), "manifest_sha256": sha256(manifest)})
    finally:
        log_handle.close(); q_handle.close()


class ImageOnlyDataset(Dataset):
    """Inference opens only the image path; it never opens annotation files."""
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        return TF.to_tensor(image), row["stem"], image.size


def collate_image(batch):
    return batch[0]


def _rank_formula(p: float, q: float, formula: str) -> float:
    if formula == "p_cls": return p
    if formula == "q": return q
    if formula == "p_cls*q": return p*q
    if formula == "p_cls*sqrt(q)": return p*math.sqrt(max(q,0.0))
    raise ValueError(formula)


def _json_quad(quad):
    return json.dumps([[float(x), float(y)] for x,y in quad], separators=(",", ":")) if quad is not None else ""


def infer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    kind = checkpoint.get("checkpoint_kind")
    quality = kind in (V9_KIND_B, "dmkpd_v8_quality_keypointrcnn")
    if kind not in (V9_KIND_A, V9_KIND_B, "dmkpd_v7_fresh_keypointrcnn", "dmkpd_v8_quality_keypointrcnn"):
        raise ValueError(f"unsupported checkpoint kind {kind}")
    is_v8 = kind == "dmkpd_v8_quality_keypointrcnn"
    is_v9_a = kind == V9_KIND_A
    score_formula = args.score_formula or ("p_cls*q" if quality else "p_cls")
    pool_formula = args.cap_formula or ("p_cls*q" if is_v8 else "p_cls")
    model = build_model(quality, candidate_budget=args.candidate_budget, pretrained_backbone=False, score_formula=score_formula, pool_formula=pool_formula).to(device)
    state = checkpoint["model"]
    model.load_state_dict(state, strict=True)
    inventory = norm_inventory(model)
    if inventory["normalization_modules"] != {"FrozenBatchNorm2d": 53}:
        raise RuntimeError(f"non-FrozenBN model constructed: {inventory['normalization_modules']}")
    model.eval()
    manifest = Path(args.manifest).resolve()
    rows = read_rows(manifest, args.role)
    if args.limit is not None:
        rows = rows[:args.limit]
    # Remove annotation paths before inference. GT is loaded separately by the
    # evaluator, never by this image-only dataset or model forward.
    image_rows = [{"image": row["image"], "stem": row["stem"]} for row in rows]
    image_dataset = ImageOnlyDataset(image_rows)
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    trace_path = output / f"{args.role}_candidate_trace.csv.gz"
    fields = ["stem","candidate_id","roi_index","rpn_proposal_count","rpn_box_xyxy","class_score","quality_score","final_score","score_formula","pool_formula","box_xyxy","raw_keypoints_json","canonical_quad_json","canonical_relation","score_pass","size_pass","box_nms_pass","prekeypoint_cap_pass","keypoint_decoded","geometry_valid","geometry_reason","polygon_nms_kept","final_output_kept","removal_reason"]
    model_ms, end_to_end_ms = [], []
    raw_invalid = canonical_fail = order_noncyclic = 0
    stage_totals = {key: 0 for key in ("rpn_proposals","score_pass","size_pass","box_nms_pass","prekeypoint_cap","geometry_valid","polygon_nms_kept","final_output")}

    # Explicit warmup is excluded from both latency distributions.
    with torch.inference_mode():
        for warmup_index in range(min(args.warmup, len(image_dataset))):
            image, _, _ = image_dataset[warmup_index]
            model([image.to(device)])
            if device.type == "cuda": torch.cuda.synchronize(device)

        with gzip.open(trace_path, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for idx, row in enumerate(rows, 1):
                t0 = time.perf_counter()
                image, stem, (width, height) = image_dataset[idx-1]
                image_device = image.to(device, non_blocking=True)
                if device.type == "cuda": torch.cuda.synchronize(device)
                mt = time.perf_counter()
                pred = model([image_device])[0]
                if device.type == "cuda": torch.cuda.synchronize(device)
                model_ms.append((time.perf_counter()-mt)*1000.0)
                debug = model.roi_heads.last_debug[0]
                pred_ids = pred["candidate_ids"].detach().cpu().long().tolist()
                points_by_id = {int(cid): pred["keypoints"][i,:,:2].detach().cpu().float().numpy().tolist() for i,cid in enumerate(pred_ids)}
                p_all = debug["p_all"].float().cpu().numpy()
                q_all = debug["q_all"].float().cpu().numpy()
                boxes_all = debug["boxes_all"].float().cpu().numpy()
                rpn_boxes = debug["rpn_proposals"].float().cpu().numpy()
                rpn_count = len(rpn_boxes)
                ids_score = set(debug["candidate_indices_after_score"].cpu().tolist())
                ids_size = set(debug["candidate_indices_after_size"].cpu().tolist())
                ids_nms = set(debug["candidate_indices_after_box_nms"].cpu().tolist())
                ids_cap = set(debug["candidate_indices_pool"].cpu().tolist())
                orig_rank_ids = pred_ids
                raw_by_id: dict[int,list[list[float]]] = {}
                quad_by_id: dict[int,list[tuple[float,float]]|None] = {}
                reason_by_id: dict[int,str] = {}
                for cand_id in ids_cap:
                    points = points_by_id.get(int(cand_id), [])
                    quad, reason = parse_prediction(points, width, height)
                    raw_by_id[int(cand_id)] = points
                    quad_by_id[int(cand_id)] = quad
                    reason_by_id[int(cand_id)] = reason

                legal_ids = [int(cid) for cid in orig_rank_ids if quad_by_id.get(int(cid)) is not None and not reason_by_id.get(int(cid))]
                ranked_legal = sorted(legal_ids, key=lambda cid: (-_rank_formula(float(p_all[cid]),float(q_all[cid]),score_formula),cid))
                polygon_keep=[]
                for cid in ranked_legal:
                    if all(quad_iou(quad_by_id[cid], quad_by_id[kept]) < args.polygon_nms for kept in polygon_keep):
                        polygon_keep.append(cid)
                final_keep = set(polygon_keep[:args.final_output_cap])
                # End-to-end timing includes image decode/transfer, network,
                # four-corner decode, geometry checks, and polygon NMS, while
                # excluding the diagnostic trace-row/JSON construction below.
                if device.type == "cuda": torch.cuda.synchronize(device)
                end_to_end_ms.append((time.perf_counter()-t0)*1000.0)
                if args.no_trace:
                    if idx == 1 or idx % args.progress_interval == 0 or idx == len(rows):
                        print(json.dumps({"inference":f"{idx}/{len(rows)}","timing_only":True,"mean_model_ms":statistics.mean(model_ms),"mean_e2e_ms":statistics.mean(end_to_end_ms)},ensure_ascii=False),flush=True)
                    continue
                poly_keep_set = set(polygon_keep)
                resized_h, resized_w = debug["image_shape_resized"]
                sx, sy = width / resized_w, height / resized_h
                image_trace_rows=[]
                for cand_id in range(len(p_all)):
                    p = float(p_all[cand_id]); q = float(q_all[cand_id])
                    final_score = _rank_formula(p,q,score_formula)
                    box = [float(boxes_all[cand_id,0])*sx,float(boxes_all[cand_id,1])*sy,float(boxes_all[cand_id,2])*sx,float(boxes_all[cand_id,3])*sy]
                    rpn_box = [float(rpn_boxes[cand_id,0])*sx,float(rpn_boxes[cand_id,1])*sy,float(rpn_boxes[cand_id,2])*sx,float(rpn_boxes[cand_id,3])*sy]
                    cap = cand_id in ids_cap
                    quad = quad_by_id.get(cand_id)
                    geo_reason = reason_by_id.get(cand_id, "") if cap else ""
                    relation = "not_decoded"
                    if cap:
                        points = raw_by_id.get(cand_id, [])
                        if quad is None:
                            relation = "invalid"
                            canonical_fail += 1
                        else:
                            # Diagnostic-only class: cyclic rotation/reversal or
                            # non-cyclic hull reordering by canonicalization.
                            def nearest_index(pt): return min(range(4), key=lambda j: math.dist(pt[:2], points[j][:2]))
                            perm = [nearest_index(pt) for pt in quad]
                            cyc = any(perm == [((s + d*k) % 4) for k in range(4)] for s in range(4) for d in (1,-1))
                            relation = "cyclic_or_reversed" if cyc else "hull_reordered"
                            if not cyc: order_noncyclic += 1
                        if quad is None or geo_reason: raw_invalid += 1
                    poly_keep = cand_id in poly_keep_set
                    out_keep = cand_id in final_keep
                    if cand_id not in ids_score: remove_reason="score_threshold"
                    elif cand_id not in ids_size: remove_reason="small_box"
                    elif cand_id not in ids_nms: remove_reason="box_nms"
                    elif not cap: remove_reason="prekeypoint_candidate_cap"
                    elif geo_reason: remove_reason=geo_reason
                    elif not poly_keep: remove_reason="polygon_nms"
                    elif not out_keep: remove_reason="final_output_cap"
                    else: remove_reason="kept"
                    stage_totals["rpn_proposals"] += 1
                    stage_totals["score_pass"] += int(cand_id in ids_score)
                    stage_totals["size_pass"] += int(cand_id in ids_size)
                    stage_totals["box_nms_pass"] += int(cand_id in ids_nms)
                    stage_totals["prekeypoint_cap"] += int(cap)
                    stage_totals["geometry_valid"] += int(cap and quad is not None and not geo_reason)
                    stage_totals["polygon_nms_kept"] += int(poly_keep)
                    stage_totals["final_output"] += int(out_keep)
                    image_trace_rows.append({
                        "stem":stem,"candidate_id":cand_id,"roi_index":cand_id,"rpn_proposal_count":rpn_count,"rpn_box_xyxy":json.dumps(rpn_box,separators=(",",":")),
                        "class_score":p,"quality_score":q if quality else "","final_score":final_score,"score_formula":score_formula,"pool_formula":pool_formula,
                        "box_xyxy":json.dumps(box,separators=(",",":")),"raw_keypoints_json":json.dumps(raw_by_id.get(cand_id,[]),separators=(",",":")),"canonical_quad_json":_json_quad(quad),"canonical_relation":relation,
                        "score_pass":int(cand_id in ids_score),"size_pass":int(cand_id in ids_size),"box_nms_pass":int(cand_id in ids_nms),"prekeypoint_cap_pass":int(cap),"keypoint_decoded":int(cap),
                        "geometry_valid":int(cap and quad is not None and not geo_reason),"geometry_reason":geo_reason,"polygon_nms_kept":int(poly_keep),"final_output_kept":int(out_keep),"removal_reason":remove_reason,
                    })
                # CSV disk writes happen after the endpoint timer by design.
                writer.writerows(image_trace_rows)
                if idx == 1 or idx % args.progress_interval == 0 or idx == len(rows):
                    handle.flush()
                    print(json.dumps({"inference":f"{idx}/{len(rows)}","trace_rows":idx,"mean_model_ms":statistics.mean(model_ms),"mean_e2e_ms":statistics.mean(end_to_end_ms)},ensure_ascii=False),flush=True)
    latency_path = output / f"{args.role}_per_image_latency.csv"
    with latency_path.open("w", encoding="utf-8", newline="") as latency_handle:
        latency_writer = csv.DictWriter(latency_handle, fieldnames=["stem", "model_ms", "end_to_end_ms"])
        latency_writer.writeheader()
        latency_writer.writerows({"stem": row["stem"], "model_ms": mm, "end_to_end_ms": ee}
                                for row, mm, ee in zip(rows, model_ms, end_to_end_ms))
    summary = {
        "role":args.role,"images":len(rows),"checkpoint":str(checkpoint_path),"checkpoint_sha256":sha256(checkpoint_path),"checkpoint_kind":kind,
        "device":str(device),"precision":"FP32","batch_size":1,"warmup_images":min(args.warmup,len(rows)),
        "quality_head":quality,"score_formula":score_formula,"prekeypoint_pool_formula":pool_formula,"prekeypoint_candidate_budget":args.candidate_budget,"final_output_cap":args.final_output_cap,"polygon_nms_iou":args.polygon_nms,
        "normalization_inventory":inventory["normalization_modules"],"normalization_trainable_parameters":[],
        "stage_total_counts":stage_totals if not args.no_trace else None,"raw_invalid_or_out_of_bounds":raw_invalid if not args.no_trace else None,"canonicalization_failures":canonical_fail if not args.no_trace else None,"noncyclic_point_reorders":order_noncyclic if not args.no_trace else None,
        "model_latency_ms":{"mean":statistics.mean(model_ms),"p50":float(np.percentile(model_ms,50)),"p95":float(np.percentile(model_ms,95))},
        "end_to_end_latency_ms":{"mean":statistics.mean(end_to_end_ms),"p50":float(np.percentile(end_to_end_ms,50)),"p95":float(np.percentile(end_to_end_ms,95)),"excludes_annotation_reads_and_csv_writes":True},
        "trace":str(trace_path),"trace_sha256":sha256(trace_path),"trace_available":not args.no_trace,"timing_only":bool(args.no_trace),"official_evaluator_available":False,"oks_definition":"local four-corner approximation; not official evaluator score",
        "per_image_latency_csv":str(latency_path),"per_image_latency_sha256":sha256(latency_path),
        "inference_source_sha256":sha256(Path(__file__).resolve()),"manifest_sha256":sha256(manifest),
        "device_name":torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU","torch_version":torch.__version__,
        "timing_protocol":"FP32 batch=1; image decode/transfer, model, canonicalization and polygon NMS; explicit warmup and CUDA synchronize; trace-row construction, annotation reads and file writes excluded",
    }
    write_json(output / f"{args.role}_inference_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def verify_checkpoint(args: argparse.Namespace) -> None:
    device=torch.device(args.device)
    checkpoint_path=Path(args.checkpoint).resolve()
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    kind=checkpoint["checkpoint_kind"]; quality=kind in (V9_KIND_B,"dmkpd_v8_quality_keypointrcnn")
    model=build_model(quality,args.candidate_budget,False,"p_cls*q" if quality else "p_cls","p_cls").to(device)
    model.load_state_dict(checkpoint["model"],strict=True); model.eval()
    manifest=Path(args.manifest).resolve(); rows=read_rows(manifest,"dev")[:args.images]
    def forward(mm,row):
        with Image.open(row["image"]) as im: image=TF.to_tensor(im.convert("RGB")).to(device)
        with torch.inference_mode(): p=mm([image])[0]
        if device.type=="cuda":torch.cuda.synchronize(device)
        return {k:v.detach().cpu() for k,v in p.items() if torch.is_tensor(v)}
    out_a=[forward(model,row) for row in rows]
    tmp=args.copy_checkpoint or str(checkpoint_path.with_name(checkpoint_path.stem+"_roundtrip.tmp.pt"))
    torch.save(checkpoint, tmp)
    loaded=torch.load(tmp,map_location="cpu",weights_only=False)
    model_b=build_model(quality,args.candidate_budget,False,"p_cls*q" if quality else "p_cls","p_cls").to(device)
    model_b.load_state_dict(loaded["model"],strict=True); model_b.eval()
    out_b=[forward(model_b,row) for row in rows]
    compare={}
    for key in ("boxes","scores","keypoints","class_scores","quality_scores"):
        deltas=[]
        for a,b in zip(out_a,out_b):
            if key in a and key in b:
                if a[key].shape!=b[key].shape: raise AssertionError(f"shape mismatch {key}: {a[key].shape} {b[key].shape}")
                deltas.append(float((a[key].float()-b[key].float()).abs().max()) if a[key].numel() else 0.0)
        compare[key]={"max_abs_difference":max(deltas,default=0.0),"identical":all(x==0.0 for x in deltas)}
    if any(not item["identical"] for item in compare.values()): raise AssertionError(compare)
    write_json(Path(args.output).resolve(),{"status":"passed","checkpoint":str(checkpoint_path),"images":len(rows),"zero_updates":True,"compare":compare,"normalization":norm_inventory(model)["normalization_modules"]})
    if not args.copy_checkpoint: Path(tmp).unlink(missing_ok=True)
    print(json.dumps({"status":"passed","compare":compare},ensure_ascii=False),flush=True)


def verify_resume_state(args: argparse.Namespace) -> None:
    """Audit a real epoch-boundary continuation bundle without updating weights."""
    path=Path(args.checkpoint).resolve();ck=torch.load(path,map_location="cpu",weights_only=False)
    epoch=int(ck["epoch"]);seed=int(ck["seed"]);manifest=Path(args.manifest).resolve()
    if ck.get("manifest_sha256")!=sha256(manifest):raise ValueError("resume manifest hash mismatch")
    if int(ck["sampler_state"]["completed_epoch"])!=epoch:raise AssertionError("sampler completed epoch mismatch")
    if int(ck["sampler_state"]["next_epoch_seed"])!=seed+epoch:raise AssertionError("next sampler seed mismatch")
    if int(ck["augmentation_state"]["completed_epoch"])!=epoch-1 or int(ck["augmentation_state"]["next_augmentation_epoch"])!=epoch:raise AssertionError("augmentation epoch mismatch")
    config=ck["registered_config"];budget=int(config["candidate_budget_pre_keypoint"])
    quality=ck["checkpoint_kind"]==V9_KIND_B
    model=build_model(quality,budget,False,"p_cls*q" if quality else "p_cls","p_cls")
    model.load_state_dict(ck["model"],strict=True);inventory=norm_inventory(model)
    if inventory["normalization_modules"]!={"FrozenBatchNorm2d":53}:raise AssertionError(inventory["normalization_modules"])
    optimizer=torch.optim.SGD((p for p in model.parameters() if p.requires_grad),lr=.0025,momentum=.9,weight_decay=.0001)
    optimizer.load_state_dict(ck["optimizer"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=12,eta_min=.00005)
    scheduler.load_state_dict(ck["scheduler"])
    if int(scheduler.last_epoch)!=epoch:raise AssertionError(f"scheduler epoch {scheduler.last_epoch} != checkpoint epoch {epoch}")
    momentum_buffers=sum("momentum_buffer" in state for state in optimizer.state.values())
    if momentum_buffers==0:raise AssertionError("optimizer has no momentum buffers")
    for state in optimizer.state.values():
        if "momentum_buffer" in state and not torch.isfinite(state["momentum_buffer"]).all():raise AssertionError("non-finite optimizer momentum")
    rows=read_rows(manifest,"train_fit");dataset=TrainDataset(rows,seed,epoch=epoch)
    aug1=dataset._aug(0);aug2=dataset._aug(0)
    if aug1!=aug2:raise AssertionError("next augmentation is not deterministic")
    g1=torch.Generator().manual_seed(seed+epoch);g2=torch.Generator().manual_seed(int(ck["sampler_state"]["next_epoch_seed"]))
    order1=torch.randperm(len(rows),generator=g1);order2=torch.randperm(len(rows),generator=g2)
    if not torch.equal(order1,order2):raise AssertionError("next sampler order mismatch")
    rng=ck["rng_state"];restore_rng(rng);restored=capture_rng()
    checks={
        "python_rng_exact":restored["python"]==rng["python"],
        "numpy_rng_exact":restored["numpy"][0]==rng["numpy"][0] and np.array_equal(restored["numpy"][1],rng["numpy"][1]) and restored["numpy"][2:]==rng["numpy"][2:],
        "torch_cpu_rng_exact":torch.equal(restored["torch"],rng["torch"]),
        "torch_cuda_rng_exact":len(restored["cuda"])==len(rng["cuda"]) and all(torch.equal(a,b) for a,b in zip(restored["cuda"],rng["cuda"])),
    }
    if not all(checks.values()):raise AssertionError(checks)
    result={"status":"passed","checkpoint":str(path),"zero_training_updates":True,"epoch_completed":epoch,"next_epoch":epoch,"next_sampler_seed":seed+epoch,"next_augmentation_epoch":epoch,"next_sample_augmentation":aug1,"optimizer_state_entries":len(optimizer.state),"momentum_buffers":momentum_buffers,"scheduler_last_epoch":scheduler.last_epoch,"scheduler_lr":scheduler.get_last_lr(),"rng_checks":checks,"normalization":inventory["normalization_modules"]}
    write_json(Path(args.output).resolve(),result);print(json.dumps(result,ensure_ascii=False),flush=True)


def smoke(args: argparse.Namespace) -> None:
    device=torch.device(args.device);seed_all(args.seed)
    model=build_model(True,candidate_budget=100,pretrained_backbone=False).to(device).train()
    if args.smoke_type=="synthetic":
        image=torch.rand((3,384,512),device=device)
        quad=[(95.0,70.0),(365.0,66.0),(371.0,302.0),(89.0,310.0)]
        xs=[p[0] for p in quad];ys=[p[1] for p in quad]
        boxes=torch.tensor([[min(xs),min(ys),max(xs),max(ys)]],dtype=torch.float32,device=device)
        target={"boxes":boxes,"labels":torch.ones((1,),dtype=torch.int64,device=device),"keypoints":torch.tensor([[[x,y,2.0] for x,y in quad]],dtype=torch.float32,device=device),"image_id":torch.tensor([1],device=device),"area":(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),"iscrowd":torch.zeros((1,),dtype=torch.int64,device=device),"gt_small":torch.tensor([False],device=device),"gt_tiny":torch.tensor([False],device=device)}
    else:
        manifest=Path(args.manifest).resolve();rows=read_rows(manifest,"train_fit")
        ds=TrainDataset(rows,args.seed,epoch=0);image,target,_=ds[0]
        image=image.to(device);target={k:v.to(device) for k,v in target.items()}
    optimizer=torch.optim.SGD(model.parameters(),lr=.0025,momentum=.9)
    losses=model([image],[target]);total=sum(losses.values())
    if not torch.isfinite(total):raise FloatingPointError("smoke loss non-finite")
    total.backward()
    qgrads=[p.grad.detach().float().norm() for p in model.roi_heads.quality_head.parameters() if p.grad is not None]
    qgrad=float(torch.stack(qgrads).norm().cpu()) if qgrads else 0.0
    if qgrad<=0 or not math.isfinite(qgrad):raise RuntimeError("quality branch gradient is zero or non-finite")
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.0)
    if not torch.isfinite(grad_norm):raise FloatingPointError("smoke gradient non-finite")
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():prediction=model([image])[0]
    h,w=image.shape[-2:];valid=0;raw_order=[]
    for points in prediction["keypoints"]:
        quad,reason=parse_prediction(points[:,:2].float().cpu().tolist(),w,h)
        valid+=int(quad is not None and not reason)
        raw_order.append({"quad":quad,"reason":reason})
    result={"status":"passed","type":args.smoke_type,"device":str(device),"total_loss":float(total.detach().cpu()),"losses":{k:float(v.detach().cpu()) for k,v in losses.items()},"global_gradient_norm":float(grad_norm.cpu()),"quality_head_gradient_norm":qgrad,"quality_stats":model.roi_heads.last_quality_stats,"forward_candidates":len(prediction["scores"]),"legal_ordered_convex_in_bounds_candidates":valid,"normalization":norm_inventory(model)["normalization_modules"],"four_corner_head":True}
    if args.output:
        write_json(Path(args.output).resolve(),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare"); q.add_argument("--output-dir",required=True); q.add_argument("--seed",type=int,default=20261009); q.add_argument("--candidate-budget",type=int,default=300); q.set_defaults(fn=prepare_initializations)
    q=sub.add_parser("train"); q.add_argument("--init-checkpoint",required=True); q.add_argument("--output-dir",required=True); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--seed",type=int,default=20261009); q.add_argument("--epochs",type=int,default=12); q.add_argument("--workers",type=int,default=2); q.add_argument("--accumulation",type=int,default=4); q.add_argument("--learning-rate",type=float,default=.0025); q.add_argument("--weight-decay",type=float,default=.0001); q.add_argument("--gradient-clip",type=float,default=10.0); q.add_argument("--amp",action="store_true"); q.add_argument("--device",default="cuda"); q.add_argument("--candidate-budget",type=int,default=300); q.add_argument("--progress-interval",type=int,default=100); q.add_argument("--resume"); q.set_defaults(fn=train)
    q=sub.add_parser("infer"); q.add_argument("--checkpoint",required=True); q.add_argument("--output-dir",required=True); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--role",default="dev"); q.add_argument("--device",default="cuda"); q.add_argument("--workers",type=int,default=0); q.add_argument("--candidate-budget",type=int,default=300); q.add_argument("--final-output-cap",type=int,default=100); q.add_argument("--polygon-nms",type=float,default=.30); q.add_argument("--warmup",type=int,default=30); q.add_argument("--progress-interval",type=int,default=100); q.add_argument("--limit",type=int); q.add_argument("--score-formula",choices=["p_cls","q","p_cls*q","p_cls*sqrt(q)"]); q.add_argument("--cap-formula",choices=["p_cls","q","p_cls*q","p_cls*sqrt(q)"]); q.add_argument("--no-trace",action="store_true"); q.set_defaults(fn=infer)
    q=sub.add_parser("verify"); q.add_argument("--checkpoint",required=True); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--output",required=True); q.add_argument("--device",default="cuda"); q.add_argument("--candidate-budget",type=int,default=300); q.add_argument("--images",type=int,default=3); q.add_argument("--copy-checkpoint"); q.set_defaults(fn=verify_checkpoint)
    q=sub.add_parser("verify-resume"); q.add_argument("--checkpoint",required=True); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--output",required=True); q.set_defaults(fn=verify_resume_state)
    q=sub.add_parser("smoke"); q.add_argument("--smoke-type",choices=["synthetic","real"],required=True); q.add_argument("--manifest",default=str(DEFAULT_MANIFEST)); q.add_argument("--seed",type=int,default=20261009); q.add_argument("--device",default="cuda"); q.add_argument("--output"); q.set_defaults(fn=smoke)
    args=p.parse_args(); args.fn(args)

if __name__=="__main__": main()
