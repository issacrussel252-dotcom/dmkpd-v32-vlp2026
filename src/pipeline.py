"""Frozen V32 candidate refinement and submission postprocessing."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from collections import Counter
import numpy as np
import torch
import v31_core as core
from v31_core import jsonable
SCORE_FLOOR = 0.30
NMS_IOU = 0.30
FINAL_CAP = 100

def run_one(model, captured, quality_head, residual_head, mean, std, image_path: Path,
            stem: str, image_sha: str | None = None, run_cfg_sha: str | None = None,
            collect_trace: bool = True, precomputed_out: dict[str, Any] | None = None) -> dict[str, Any]:
    start_e2e = time.perf_counter()
    torch.cuda.synchronize()
    out = precomputed_out if precomputed_out is not None else core.run_image(model, captured, quality_head, mean, std, image_path)
    n = len(out["candidate_ids"])
    if n > 100:
        raise RuntimeError(f"pre-keypoint candidate budget exceeded on {stem}: {n}")
    device = next(residual_head.parameters()).device
    spatial = torch.from_numpy(out["spatial_features"]).to(device=device, dtype=torch.float32)
    coords = torch.from_numpy(out["roi_corner_norm"]).to(device=device, dtype=torch.float32)
    residual_start = torch.cuda.Event(enable_timing=True)
    residual_end = torch.cuda.Event(enable_timing=True)
    residual_start.record()
    with torch.inference_mode():
        z = residual_head(spatial, coords).reshape(-1, 4, 2)
        roi = torch.from_numpy(out["rois_resized"]).to(device=device, dtype=torch.float32)
        delta = core.decode_residual(z, roi, out["original_size"], out["resized_size"])
    residual_end.record(); residual_end.synchronize()
    residual_ms = float(residual_start.elapsed_time(residual_end))
    z_np = z.detach().cpu().numpy().astype(np.float64)
    delta_np = delta.detach().cpu().numpy().astype(np.float64)
    raw_v28 = np.asarray(out["raw_v28"], dtype=np.float64)
    proposal = raw_v28 + delta_np
    width, height = map(int, out["original_size"])
    pool = {stem: []}
    candidates: list[dict[str, Any]] = []
    no_residual_pool: list[dict[str, Any]] = []
    decision_counts = Counter()
    invalid_raw = invalid_proposal = invalid_final = 0
    for i, cid0 in enumerate(out["candidate_ids"]):
        cid = int(cid0)
        raw_n = np.asarray(out["raw_native"][i], dtype=np.float64)
        base = raw_v28[i]
        prop = proposal[i]
        base_quad, base_err, base_perm, base_ok = core.canonical_with_permutation(base, width, height)
        prop_quad, prop_err, prop_perm, prop_ok = core.canonical_with_permutation(prop, width, height)
        disabled, disabled_decision, _disabled_reasons = core.guard_candidate(
            base, base.copy(), width, height, False)
        if not np.array_equal(disabled, base):
            raise RuntimeError(f"zero-residual/disabled branch differs from V28: {stem}/{cid}")
        triggered = bool(np.any(delta_np[i] != 0.0))
        final_raw, decision, reasons = core.guard_candidate(base, prop, width, height, triggered)
        final_quad, final_err, final_perm, final_ok = core.canonical_with_permutation(final_raw, width, height)
        base_legal = bool(base_ok and not base_err)
        prop_legal = bool(prop_ok and not prop_err)
        final_legal = bool(final_ok and not final_err)
        invalid_raw += int(not base_legal)
        invalid_proposal += int(not prop_legal)
        invalid_final += int(not final_legal)
        if not base_legal and final_legal:
            raise RuntimeError(f"V31 guard unexpectedly rescued invalid V28 candidate {stem}/{cid}")
        if base_legal and not final_legal:
            raise RuntimeError(f"V31 guard emitted invalid candidate {stem}/{cid}")
        decision_counts[decision] += 1
        for reason in reasons:
            decision_counts[f"reason:{reason}"] += 1
        score = float(out["quality_scores"][i])
        cls = float(out["class_scores"][i])
        if base_legal:
            no_residual_pool.append({"candidate_id": cid, "score": score,
                                     "quality_score": score, "class_score": cls,
                                     "quad": base_quad})
        if final_legal:
            pool[stem].append({"candidate_id": cid, "score": score,
                               "quality_score": score, "class_score": cls,
                               "quad": final_quad})
        candidates.append({
            "image_id": stem, "candidate_id": cid,
            "box_xyxy_original": np.asarray(out["boxes_original"][i], dtype=np.float64).tolist(),
            "box_xyxy_resized": np.asarray(out["boxes_resized"][i], dtype=np.float64).tolist(),
            "class_score": cls,
            "quality_score": score,
            "final_score": score,
            "score_semantics": "V10 quality ranking score; not a calibrated localization probability",
            "raw_native_corners": raw_n.tolist(),
            "V28_corners": base.tolist(),
            "residual_proposal_corners": prop.tolist(),
            "final_raw_corners": final_raw.tolist(),
            "V28_canonical_corners": jsonable(base_quad),
            "proposal_canonical_corners": jsonable(prop_quad),
            "final_canonical_corners": jsonable(final_quad),
            "normalized_residual_prediction_z": z_np[i].tolist(),
            "original_pixel_residual_xy": delta_np[i].tolist(),
            "V28_permutation_canonical_to_raw": list(base_perm) if base_perm is not None else None,
            "proposal_permutation_canonical_to_raw": list(prop_perm) if prop_perm is not None else None,
            "final_permutation_canonical_to_raw": list(final_perm) if final_perm is not None else None,
            "V28_legal": base_legal, "proposal_legal": prop_legal, "final_legal": final_legal,
            "V28_geometry_error": base_err, "proposal_geometry_error": prop_err,
            "final_geometry_error": final_err,
            "residual_decision": decision, "residual_reasons": reasons,
            "V28_fit_state_by_corner": np.asarray(out["fit_state"][i]).astype(int).tolist(),
            "V28_fit_offset_resized": np.asarray(out["fit_offset_resized"][i]).tolist(),
            "geometry_filter_kept": final_legal,
            "polygon_nms_kept": False, "polygon_nms_suppressor_candidate_id": None,
            "polygon_nms_iou_with_suppressor": None,
            "final_cap_kept": False, "score_floor_kept": False,
            "removal_stage": None,
        })

    # Primary postprocessing delegates to the frozen evaluator implementation.
    final_map = core.ev.rerank_and_nms(pool, "q", NMS_IOU, FINAL_CAP)
    no_residual_map = core.ev.rerank_and_nms({stem: no_residual_pool}, "q", NMS_IOU, FINAL_CAP)
    final_rows = final_map.get(stem, [])
    a_ids = {int(r["candidate_id"]) for r in no_residual_map.get(stem, [])}
    no_floor_ids = [int(r["candidate_id"]) for r in final_rows]
    no_floor_by_id = {int(r["candidate_id"]): r for r in final_rows}
    # Reconstruct the same ranked NMS sequence to preserve suppression reasons.
    ranked = sorted(pool[stem], key=lambda x: (-float(x["quality_score"]), int(x["candidate_id"])))
    kept_all: list[dict[str, Any]] = []
    suppressed: dict[int, tuple[int, float]] = {}
    for row in ranked:
        suppressor = None
        overlap_value = None
        for winner in kept_all:
            overlap = float(core.ev.quad_iou(row["quad"], winner["quad"]))
            if overlap >= NMS_IOU:
                suppressor = int(winner["candidate_id"])
                overlap_value = overlap
                break
        if suppressor is None:
            kept_all.append(row)
        else:
            suppressed[int(row["candidate_id"])] = (suppressor, float(overlap_value))
    kept_all_ids = {int(r["candidate_id"]) for r in kept_all}
    capped_ids = set(no_floor_ids)
    floor_rows = [r for r in final_rows if float(r["score"]) >= SCORE_FLOOR]
    floor_ids = {int(r["candidate_id"]) for r in floor_rows}
    # Ensure the delegated evaluator and detailed stage accounting agree exactly.
    if no_floor_ids != [int(r["candidate_id"]) for r in kept_all[:FINAL_CAP]]:
        raise RuntimeError(f"detailed NMS trace differs from frozen evaluator for {stem}")
    row_by_id = {r["candidate_id"]: r for r in candidates}
    for r in candidates:
        cid = int(r["candidate_id"])
        if cid in suppressed:
            r["polygon_nms_suppressor_candidate_id"], r["polygon_nms_iou_with_suppressor"] = suppressed[cid]
            r["removal_stage"] = "polygon_nms"
        elif cid in kept_all_ids:
            r["polygon_nms_kept"] = True
            if cid not in capped_ids:
                r["removal_stage"] = "final_output_cap_100"
        elif not r["final_legal"]:
            r["removal_stage"] = "geometry_filter"
        else:
            r["removal_stage"] = "unclassified"
            raise RuntimeError(f"candidate not accounted for after NMS: {stem}/{cid}")
        r["final_cap_kept"] = cid in capped_ids
        r["score_floor_kept"] = cid in floor_ids
        r["NMS_A_kept"] = cid in a_ids
        r["NMS_B_kept"] = cid in capped_ids
        r["score_ge_0_30"] = float(r["quality_score"]) >= SCORE_FLOOR
        if cid in capped_ids and cid not in floor_ids:
            r["removal_stage"] = "score_floor_0_30"
    final_predictions = []
    for rank_no, row in enumerate(floor_rows, 1):
        cid = int(row["candidate_id"])
        src = row_by_id[cid]
        final_predictions.append({
            "image_id": stem, "rank": rank_no, "candidate_id": cid,
            "class_score": src["class_score"], "quality_score": src["quality_score"],
            "final_score": float(row["score"]),
            "quad": np.asarray(row["quad"], dtype=np.float64).tolist(),
        })
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - start_e2e) * 1000.0
    model_gpu_ms = float(out["model_gpu_ms"] + out["decoder_gpu_ms"] + out["quality_and_spatial_gpu_ms"] + residual_ms)
    return {
        "image_id": stem, "image_path": str(image_path), "image_sha256": image_sha,
        "run_config_sha256": run_cfg_sha,
        "original_size_wh": list(out["original_size"]),
        "resized_size_hw": list(out["resized_size"]),
        "resize_mapping": {"scale_x": out["resized_size"][1] / out["original_size"][0],
                            "scale_y": out["resized_size"][0] / out["original_size"][1],
                            "original_width": out["original_size"][0],
                            "original_height": out["original_size"][1],
                            "resized_width": out["resized_size"][1],
                            "resized_height": out["resized_size"][0]},
        "candidate_count": n, "candidates": candidates,
        "predictions_no_score_floor": [
            {"candidate_id": int(r["candidate_id"]), "score": float(r["score"]),
             "quad": np.asarray(r["quad"], dtype=np.float64).tolist()}
            for r in final_rows],
        "predictions_score_floor_0_30": final_predictions,
        "counts": {"raw_invalid": invalid_raw, "proposal_invalid": invalid_proposal,
                   "final_invalid_before_geometry_filter": invalid_final,
                   "geometry_valid": sum(bool(r["final_legal"]) for r in candidates),
                   "polygon_nms_kept_before_cap": len(kept_all),
                   "polygon_nms_suppressed": len(suppressed),
                   "final_cap_kept": len(no_floor_ids), "score_floor_kept": len(floor_ids),
                   "empty_prediction_image": int(len(final_predictions) == 0)},
        "residual_decisions": dict(decision_counts),
        "timing_ms": {"detector_gpu": float(out["model_gpu_ms"]),
                      "quality_and_spatial_gpu": float(out["quality_and_spatial_gpu_ms"]),
                      "V28_decode_gpu": float(out["decoder_gpu_ms"]),
                      "V32_residual_head_and_decode_gpu": residual_ms,
                      "model_pipeline_gpu_sum": model_gpu_ms,
                      "end_to_end_in_memory": e2e_ms},
    }
