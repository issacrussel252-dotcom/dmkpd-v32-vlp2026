from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from v31_core import (BASE, BASE_SHA, HEAD_CKPT, HEAD_SHA, GRID, MANIFEST, MAX_CANDIDATES, ROOT, V28,
                      append_jsonl, canonical_with_permutation, dump_json,
                      jsonable, load_models, run_image, sha256)


FIELDS = {
    "features": ((256, 4, 4), np.float32),
    "candidate_ids": ((), np.int32),
    "class_scores": ((), np.float32),
    "quality_scores": ((), np.float32),
    "boxes_original": ((4,), np.float32),
    "boxes_resized": ((4,), np.float32),
    "raw_native": ((4, 2), np.float64),
    "raw_v28": ((4, 2), np.float64),
    "raw_v28_online": ((4, 2), np.float64),
    "rois_resized": ((4,), np.float32),
    "corner_norm": ((4, 2), np.float32),
    "valid": ((), np.uint8),
    "permutation": ((4,), np.int8),
    "fit_state": ((4,), np.int8),
    "fit_offset_resized": ((4, 2), np.float64),
    "image_size": ((2,), np.int32),
    "resized_size": ((2,), np.int32),
    "candidate_count": ((), np.int16),
    "max_iou": ((), np.float32),
    "gt_index": ((), np.int32),
    "group": ((), np.int8),
    "ignore_reason": ((), np.int8),
    "target_norm": ((4, 2), np.float32),
    "matched_small": ((), np.uint8),
    "matched_tiny": ((), np.uint8),
}

REASON = {
    0: "foreground",
    1: "background",
    2: "iou_0.1_to_0.5_ignored",
    3: "v28_invalid_ignored",
    4: "foreground_best_gt_tie_ignored",
    5: "unreliable_corner_identity_ignored",
}


def open_maps(directory: Path, image_count: int, create: bool):
    directory.mkdir(parents=True, exist_ok=True)
    maps = {}
    for name, (tail_shape, dtype) in FIELDS.items():
        shape = (image_count, MAX_CANDIDATES, *tail_shape)
        path = directory / f"{name}.npy"
        if create:
            maps[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            if name in ("candidate_ids", "gt_index"):
                maps[name][:] = -1
            elif name == "permutation":
                maps[name][:] = -1
            elif name == "group":
                maps[name][:] = -1
            elif name == "ignore_reason":
                maps[name][:] = 5
            elif name == "max_iou":
                maps[name][:] = np.nan
            elif name == "target_norm":
                maps[name][:] = 0
            elif name in ("valid", "candidate_count", "matched_small", "matched_tiny"):
                maps[name][:] = 0
        else:
            if not path.exists():
                raise FileNotFoundError(path)
            maps[name] = np.load(path, mmap_mode="r+")
            if maps[name].shape != shape or maps[name].dtype != dtype:
                raise RuntimeError(f"cache shape/dtype mismatch for {path}: {maps[name].shape} {maps[name].dtype}")
    return maps


def assign_train_targets(record: dict, gt_quads: list, width: int, height: int):
    """Return fixed-IoU group and raw-channel target without permutation search."""
    n = len(record["candidate_ids"])
    group = np.full(n, -1, dtype=np.int8)
    reason = np.full(n, 5, dtype=np.int8)
    max_iou = np.full(n, np.nan, dtype=np.float32)
    gt_index = np.full(n, -1, dtype=np.int32)
    target_norm = np.zeros((n, 4, 2), dtype=np.float32)
    matched_small = np.zeros(n, dtype=np.uint8)
    matched_tiny = np.zeros(n, dtype=np.uint8)
    scale = np.asarray([record["resized_size"][1] / width,
                        record["resized_size"][0] / height], dtype=np.float64)
    rois = record["rois_resized"].astype(np.float64)
    roi_wh = np.maximum(rois[:, 2:4] - rois[:, 0:2], 1.0)
    base = record["raw_v28"]
    quads = []
    for q in gt_quads:
        q_array = np.asarray(q, dtype=np.float64)
        if q_array.shape != (4, 2) or not np.isfinite(q_array).all():
            continue
        if not __import__("v9_geometry").ordered_quad_is_valid(q_array.tolist()):
            continue
        quads.append(q_array)

    for j in range(n):
        q, _error, perm, is_valid = canonical_with_permutation(base[j], width, height)
        if not is_valid:
            reason[j] = 3
            continue
        if perm is None:
            reason[j] = 5
            continue
        overlaps = np.asarray([__import__("v9_geometry").quad_iou(q, gt) for gt in quads], dtype=np.float64)
        if not len(overlaps):
            group[j] = 2
            reason[j] = 1
            max_iou[j] = 0.0
            continue
        order = np.argsort(-overlaps, kind="stable")
        best = float(overlaps[order[0]])
        max_iou[j] = best
        if best >= 0.5:
            if len(order) > 1 and abs(best - float(overlaps[order[1]])) <= 1e-10:
                reason[j] = 4
                continue
            gt_idx = int(order[0])
            gt = quads[gt_idx]
            # The evaluator's canonical order defines correspondence. The
            # candidate canonical->raw mapping is explicit; no GT permutation
            # is searched or selected by pointwise loss.
            for canonical_index, raw_index in enumerate(perm):
                delta_resized = (gt[canonical_index] - base[j, raw_index]) * scale
                target_norm[j, raw_index] = delta_resized / roi_wh[j]
            group[j] = 1
            reason[j] = 0
            gt_index[j] = gt_idx
            area = abs(float(__import__("v9_geometry").polygon_area(gt.tolist())))
            matched_small[j] = int(area / max(width * height, 1) < 0.01)
            matched_tiny[j] = int(min(np.linalg.norm(gt[k] - gt[(k + 1) % 4]) for k in range(4)) < 16.0)
        elif best < 0.1:
            group[j] = 2
            reason[j] = 1
        else:
            reason[j] = 2
    return {"group": group, "ignore_reason": reason, "max_iou": max_iou,
            "gt_index": gt_index, "target_norm": target_norm,
            "matched_small": matched_small, "matched_tiny": matched_tiny}


def dump_candidate_trace(directory: Path, rows, maps, role: str):
    path = directory / f"{role}_candidate_trace.csv.gz"
    fields = ["stem", "candidate_id", "class_score", "quality_score_online", "boxes_original_json", "boxes_resized_json",
              "raw_native_json", "V28_raw_index_quad_json", "V28_canonical_quad_json",
              "V28_canonical_to_raw_permutation", "roi_resized_json", "corner_norm_roi_json",
              "geometry_valid", "fit_state_json", "fit_offset_resized_json", "assignment",
              "max_polygon_iou", "matched_gt_index", "ignore_reason", "target_norm_json",
              "matched_small", "matched_tiny"]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for image_idx, row in enumerate(rows):
            count = int(maps["candidate_count"][image_idx, 0])
            width, height = map(int, maps["image_size"][image_idx, 0])
            for j in range(count):
                raw = maps["raw_v28"][image_idx, j]
                canonical, _error, perm, valid = canonical_with_permutation(raw, width, height)
                group = int(maps["group"][image_idx, j])
                writer.writerow({
                    "stem": row["stem"], "candidate_id": int(maps["candidate_ids"][image_idx, j]),
                    "class_score": float(maps["class_scores"][image_idx, j]),
                    "quality_score_online": float(maps["quality_scores"][image_idx, j]),
                    "boxes_original_json": json.dumps(maps["boxes_original"][image_idx, j].tolist(), separators=(",", ":")),
                    "boxes_resized_json": json.dumps(maps["boxes_resized"][image_idx, j].tolist(), separators=(",", ":")),
                    "raw_native_json": json.dumps(maps["raw_native"][image_idx, j].tolist(), separators=(",", ":")),
                    "V28_raw_index_quad_json": json.dumps(raw.tolist(), separators=(",", ":")),
                    "V28_canonical_quad_json": json.dumps(canonical, separators=(",", ":")) if valid else "",
                    "V28_canonical_to_raw_permutation": json.dumps(perm) if perm is not None else "",
                    "roi_resized_json": json.dumps(maps["rois_resized"][image_idx, j].tolist(), separators=(",", ":")),
                    "corner_norm_roi_json": json.dumps(maps["corner_norm"][image_idx, j].tolist(), separators=(",", ":")),
                    "geometry_valid": int(maps["valid"][image_idx, j]),
                    "fit_state_json": json.dumps(maps["fit_state"][image_idx, j].tolist()),
                    "fit_offset_resized_json": json.dumps(maps["fit_offset_resized"][image_idx, j].tolist(), separators=(",", ":")),
                    "assignment": ("foreground" if group == 1 else "background" if group == 2 else "ignored" if role == "train_fit" else "not_applied"),
                    "max_polygon_iou": "" if not np.isfinite(maps["max_iou"][image_idx, j]) else float(maps["max_iou"][image_idx, j]),
                    "matched_gt_index": int(maps["gt_index"][image_idx, j]),
                    "ignore_reason": REASON.get(int(maps["ignore_reason"][image_idx, j]), "not_applied"),
                    "target_norm_json": json.dumps(maps["target_norm"][image_idx, j].tolist(), separators=(",", ":")),
                    "matched_small": int(maps["matched_small"][image_idx, j]),
                    "matched_tiny": int(maps["matched_tiny"][image_idx, j]),
                })
    return path


def collect(role: str, pilot_limit: int | None = None):
    if role not in ("train_fit", "dev"):
        raise ValueError("only train_fit and dev feature extraction are allowed")
    if sha256(BASE) != BASE_SHA or sha256(HEAD_CKPT) != HEAD_SHA:
        raise RuntimeError("frozen model SHA256 mismatch")
    rows = __import__("v10_rank").read_rows(MANIFEST, role)
    expected = {"train_fit": 7731, "dev": 859}[role]
    if len(rows) != expected:
        raise RuntimeError(f"unexpected {role} manifest row count: {len(rows)}")
    if pilot_limit is not None:
        rows = rows[:pilot_limit]
    cache_dir = ROOT / "cache" / role
    complete_path = cache_dir / "collection_complete.json"
    if complete_path.exists():
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("manifest_sha256") == sha256(MANIFEST):
            print(json.dumps({"stage": "collect", "role": role, "status": "already_complete", "images": len(rows)}), flush=True)
            return
    if pilot_limit is not None:
        raise ValueError("pilot collection is diagnostic only; use --role with full split")
    maps = open_maps(cache_dir, len(rows), create=not (cache_dir / "features.npy").exists())
    progress_path = cache_dir / "collection_progress.json"
    progress = {"next_image_idx": 0}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    start_idx = int(progress["next_image_idx"])
    if not 0 <= start_idx <= len(rows):
        raise RuntimeError("invalid feature-cache resume cursor")
    model, quality_head, mean, std = load_models()
    captured = __import__("v10_rank").frozen_model  # retain import identity in provenance
    del captured
    # Reinstall hooks only once on this process's fresh detector.
    from v31_core import install_capture
    cap = install_capture(model)
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from v9_train import state_digest
    frozen_state_before = state_digest(model.state_dict())
    with Image.open(rows[0]["image"]) as im:
        warm = to_tensor(im.convert("RGB")).cuda()
    warm_start = torch.cuda.Event(enable_timing=True)
    warm_end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        warm_start.record()
        for _ in range(5):
            model([warm])
        warm_end.record()
        warm_end.synchronize()
    warmup_gpu_seconds = float(warm_start.elapsed_time(warm_end)) / 1000.0
    del warm
    started = time.perf_counter()
    total_model_s = float(progress.get("total_model_seconds", 0.0))
    total_extract_s = float(progress.get("total_extract_seconds", 0.0))
    v28_checks = []
    train_counts = {"foreground": 0, "background": 0, "iou_0.1_to_0.5_ignored": 0,
                    "v28_invalid_ignored": 0, "foreground_best_gt_tie_ignored": 0,
                    "unreliable_corner_identity_ignored": 0}
    for image_idx in range(start_idx, len(rows)):
        row = rows[image_idx]
        stage_start = time.perf_counter()
        result = run_image(model, cap, quality_head, mean, std, row["image"])
        count = len(result["candidate_ids"])
        if count > MAX_CANDIDATES:
            raise RuntimeError(f"candidate budget exceeded for {row['stem']}: {count}")
        if count == 0:
            raise RuntimeError(f"no candidates for {row['stem']} under frozen V28 top-100 path")
        width, height = result["original_size"]
        if role == "dev":
            reference_path = V28 / "cache" / "dev" / f"{row['stem']}.npz"
            if not reference_path.exists():
                raise FileNotFoundError(reference_path)
            with np.load(reference_path, allow_pickle=False) as ref:
                ref_ids = ref["ids"].astype(np.int32)
                if not np.array_equal(result["candidate_ids"], ref_ids):
                    raise RuntimeError(f"V28 candidate IDs mismatch at {row['stem']}")
                box_error = float(np.max(np.abs(result["boxes_original"] - ref["boxes"])))
                native_error = float(np.max(np.abs(result["raw_native"] - ref["raw_a"])))
                v28_error = float(np.max(np.abs(result["raw_v28"] - ref["raw_b"])))
                q_error = float(np.max(np.abs(result["quality_scores"] - ref["q_online"])))
                q_legacy_error = float(np.max(np.abs(result["quality_scores"] - ref["q"])))
                fit_state_equal = bool(np.array_equal(result["fit_state"], ref["fit_state"]))
                if box_error > 1e-4 or native_error > 1e-4 or not fit_state_equal:
                    raise RuntimeError(f"V28 geometry replay mismatch at {row['stem']}: {box_error}, {native_error}, {v28_error}")
                # The V28 record documented online score rounding drift; all
                # current q differences are retained, never hidden by a tolerance.
                base_for_eval = ref["raw_b"].astype(np.float64)
                v28_checks.append({"stem": row["stem"], "same_ids": True, "box_max_abs": box_error,
                                   "native_max_abs": native_error, "v28_max_abs": v28_error,
                                   "fit_state_equal": fit_state_equal,
                                   "q_vs_previous_online_max_abs": q_error,
                                   "q_vs_frozen_cache_max_abs": q_legacy_error})
        else:
            base_for_eval = result["raw_v28"].astype(np.float64)

        for name, value in (("candidate_ids", result["candidate_ids"]),
                            ("class_scores", result["class_scores"]),
                            ("quality_scores", result["quality_scores"]),
                            ("boxes_original", result["boxes_original"]),
                            ("boxes_resized", result["boxes_resized"]),
                            ("raw_native", result["raw_native"]),
                            ("raw_v28_online", result["raw_v28"]),
                            ("raw_v28", base_for_eval),
                            ("rois_resized", result["rois_resized"]),
                            ("fit_state", result["fit_state"]),
                            ("fit_offset_resized", result["fit_offset_resized"])):
            maps[name][image_idx, :count] = value
        maps["features"][image_idx, :count] = result["spatial_features"]
        maps["corner_norm"][image_idx, :count] = __import__("v31_core").roi_corner_coordinates(
            base_for_eval, result["rois_resized"], result["original_size"], result["resized_size"])
        maps["image_size"][image_idx, :count] = np.asarray([width, height], dtype=np.int32)
        maps["resized_size"][image_idx, :count] = np.asarray(result["resized_size"], dtype=np.int32)
        maps["candidate_count"][image_idx, :count] = count
        for j in range(count):
            _q, _err, perm, valid = canonical_with_permutation(base_for_eval[j], width, height)
            maps["valid"][image_idx, j] = int(valid and _err == "")
            if perm is not None:
                maps["permutation"][image_idx, j] = np.asarray(perm, dtype=np.int8)

        if role == "train_fit":
            # Annotations are opened only after the image-only detector forward.
            gt_quads = __import__("v9_geometry").parse_shapes(Path(row["annotation"]), row["stem"])
            target_record = {**result, "raw_v28": result["raw_v28_online"]}
            labels = assign_train_targets(target_record, gt_quads, width, height)
            for name, value in labels.items():
                maps[name][image_idx, :count] = value
            train_counts["foreground"] += int(np.sum(labels["group"] == 1))
            train_counts["background"] += int(np.sum(labels["group"] == 2))
            train_counts["iou_0.1_to_0.5_ignored"] += int(np.sum(labels["ignore_reason"] == 2))
            train_counts["v28_invalid_ignored"] += int(np.sum(labels["ignore_reason"] == 3))
            train_counts["foreground_best_gt_tie_ignored"] += int(np.sum(labels["ignore_reason"] == 4))
            train_counts["unreliable_corner_identity_ignored"] += int(np.sum(labels["ignore_reason"] == 5))
        torch.cuda.synchronize()
        image_seconds = time.perf_counter() - stage_start
        total_extract_s += image_seconds
        total_model_s += (result["model_gpu_ms"] + result["decoder_gpu_ms"] + result["quality_and_spatial_gpu_ms"]) / 1000.0
        if (image_idx + 1) % 25 == 0 or image_idx + 1 == len(rows):
            for mmap in maps.values():
                mmap.flush()
            progress_payload = {"next_image_idx": image_idx + 1, "total_model_seconds": total_model_s,
                                "total_extract_seconds": total_extract_s,
                                "last_stem": row["stem"], "role": role}
            dump_json(progress_path, progress_payload)
            append_jsonl(ROOT / "logs" / "execution.jsonl", {
                "stage": "feature_collection", "role": role, "images_done": image_idx + 1,
                "images_total": len(rows), "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
                "last_image": row["stem"], "last_image_seconds": round(image_seconds, 4),
                "train_counts_so_far": train_counts if role == "train_fit" else None,
            })
            print(json.dumps({"stage": "feature_collection", "role": role, "images": f"{image_idx + 1}/{len(rows)}",
                              "last_image": row["stem"], "last_seconds": round(image_seconds, 3),
                              "counts": train_counts if role == "train_fit" else None}, ensure_ascii=False), flush=True)
    for mmap in maps.values():
        mmap.flush()
    frozen_state_after = state_digest(model.state_dict())
    if frozen_state_before != frozen_state_after:
        raise RuntimeError("frozen detector state changed during collection")
    if role == "train_fit":
        train_counts = {
            "foreground": int(np.sum(maps["group"] == 1)),
            "background": int(np.sum(maps["group"] == 2)),
            "iou_0.1_to_0.5_ignored": int(np.sum(maps["ignore_reason"] == 2)),
            "v28_invalid_ignored": int(np.sum(maps["ignore_reason"] == 3)),
            "foreground_best_gt_tie_ignored": int(np.sum(maps["ignore_reason"] == 4)),
            "unreliable_corner_identity_ignored": int(np.sum(maps["ignore_reason"] == 5)),
        }
    dump_json(cache_dir / "manifest_rows.json", [{"stem": r["stem"], "image": r["image"],
                                                  "manifest_image_sha256": r.get("sha256", "")} for r in rows])
    if role == "dev":
        dump_json(cache_dir / "v28_online_equivalence.json", {
            "images": len(v28_checks), "checks": v28_checks,
            "max_box_abs": max(x["box_max_abs"] for x in v28_checks),
            "max_native_abs": max(x["native_max_abs"] for x in v28_checks),
            "max_v28_abs": max(x["v28_max_abs"] for x in v28_checks),
            "max_quality_vs_previous_online": max(x["q_vs_previous_online_max_abs"] for x in v28_checks),
            "max_quality_vs_frozen_cache": max(x["q_vs_frozen_cache_max_abs"] for x in v28_checks),
            "metric_role": "feature extraction only; no dev labels opened"})
    if role == "train_fit":
        from v31_core import geom
        dump_json(cache_dir / "supervision_summary.json", {
            "status": "complete", "images": len(rows), "candidate_slots": len(rows) * MAX_CANDIDATES,
            "group_counts": {"foreground": int(np.sum(maps["group"] == 1)),
                             "background": int(np.sum(maps["group"] == 2)),
                             "ignored": int(np.sum(maps["group"] < 0))},
            "ignore_reason_counts": {label: int(np.sum(maps["ignore_reason"] == code)) for code, label in REASON.items()},
            "foreground_small": int(np.sum((maps["group"] == 1) & (maps["matched_small"] == 1))),
            "foreground_tiny": int(np.sum((maps["group"] == 1) & (maps["matched_tiny"] == 1))),
            "foreground_target_norm_abs": {"mean": float(np.mean(np.abs(maps["target_norm"][maps["group"] == 1]))),
                                            "p95": float(np.percentile(np.abs(maps["target_norm"][maps["group"] == 1]), 95)),
                                            "max": float(np.max(np.abs(maps["target_norm"][maps["group"] == 1])))},
            "original_invalid_candidates": int(np.sum(maps["valid"] == 0)),
            "candidate_count_distribution": {str(x): int(np.sum(maps["candidate_count"][:, 0] == x)) for x in np.unique(maps["candidate_count"][:, 0])},
            "manifest_sha256": sha256(MANIFEST), "detector_sha256": BASE_SHA, "quality_head_sha256": HEAD_SHA,
        })
    trace_path = dump_candidate_trace(cache_dir, rows, maps, role)
    summary = {
        "status": "complete", "role": role, "images": len(rows), "candidate_slots": len(rows) * MAX_CANDIDATES,
        "image_forward_and_decode_seconds": total_model_s, "warmup_gpu_seconds": warmup_gpu_seconds,
        "frozen_detector_state_sha256_before": frozen_state_before,
        "frozen_detector_state_sha256_after": frozen_state_after,
        "wall_seconds": total_extract_s,
        "manifest_sha256": sha256(MANIFEST), "detector_sha256": BASE_SHA, "quality_head_sha256": HEAD_SHA,
        "feature_shape_per_candidate": [256, GRID, GRID], "feature_dtype": "float32",
        "feature_cache_bytes": int((cache_dir / "features.npy").stat().st_size),
        "trace_path": str(trace_path), "train_counts": train_counts if role == "train_fit" else None,
        "test300_read": False, "annotations_read": role == "train_fit",
    }
    dump_json(complete_path, summary)
    print(json.dumps({"stage": "collection_complete", **summary}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("train_fit", "dev"), required=True)
    args = parser.parse_args()
    collect(args.role)


if __name__ == "__main__":
    main()
