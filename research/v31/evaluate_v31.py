from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from v31_core import (BASE, BASE_SHA, GRID, HEAD_CKPT, HEAD_SHA, MANIFEST,
                      MAX_CANDIDATES, ROOT, V28, CornerResidualHead,
                      canonical_with_permutation, decode_residual, dump_json, guard_candidate,
                      finite_stats, jsonable, load_models, prediction_from_pool,
                      roi_corner_coordinates, run_image, safe_torch_load, sha256)

import v10_rank as rank
import v9_evaluate as ev
import v9_geometry as geom
import metric_helpers as mh


EXPECTED_A = {
    "local_oks_ap50_95_post_polygon_nms_no_floor": 0.7905144222533753,
    "polygon_ap50_95_post_polygon_nms_no_floor": 0.8181647489075556,
    "floor030": {"tp": 1767, "fp": 198, "fn": 214,
                 "small_detected": 612, "small_targets": 699,
                 "tiny_detected": 20, "tiny_targets": 31,
                 "boundary_detected": 23, "boundary_targets": 26},
}
V28_CACHED = V28 / "cache" / "dev"
_EVAL_STATE = None


def write_csv_gz(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        return
    fields = fields or list(rows[0])
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(jsonable(v), separators=(",", ":"), ensure_ascii=False)
                                 if isinstance(v, (list, tuple, dict, np.ndarray)) else jsonable(v))
                             for k, v in row.items()})


def load_v28_samples() -> dict[str, dict]:
    return ev.load_dev(MANIFEST, "dev")


def load_v28_pool(samples: dict[str, dict]):
    pool = {stem: [] for stem in samples}
    trace = {}
    invalid = Counter()
    for stem, sample in samples.items():
        cache_path = V28_CACHED / f"{stem}.npz"
        with np.load(cache_path, allow_pickle=False) as z:
            count = len(z["ids"])
            width, height = map(int, z["size"])
            for j in range(count):
                cid = int(z["ids"][j])
                raw = z["raw_b"][j].astype(np.float64)
                q, error = rank.parse_prediction(raw.tolist(), width, height)
                valid = q is not None and error == ""
                invalid["candidate_count"] += 1
                if not valid:
                    invalid[error or "invalid"] += 1
                row = {"stem": stem, "candidate_id": cid, "class_score": float(z["p"][j]),
                       "quality_score": float(z["q"][j]), "score": float(z["q"][j]),
                       "raw_v28": raw, "v28_quad": q, "v28_valid": bool(valid),
                       "v28_error": error, "class_probability": float(z["p"][j])}
                trace[(stem, cid)] = row
                if valid:
                    pool[stem].append({"candidate_id": cid, "class_score": float(z["p"][j]),
                                       "quality_score": float(z["q"][j]), "score": float(z["q"][j]),
                                       "quad": q})
    predictions = prediction_from_pool(pool)
    return predictions, trace, invalid


def load_cached_v28_reference():
    path = V28 / "results" / "B_predictions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare_predictions(actual, expected):
    if set(actual) != set(expected):
        return {"same_stems": False, "stem_count_actual": len(actual), "stem_count_expected": len(expected)}
    count_a = count_e = 0
    max_quad = max_score = 0.0
    different_ids = []
    for stem in sorted(actual):
        a = actual[stem]
        e = expected[stem]
        count_a += len(a); count_e += len(e)
        if len(a) != len(e):
            different_ids.append(stem)
            continue
        for ra, re in zip(a, e):
            if int(ra["candidate_id"]) != int(re["candidate_id"]):
                different_ids.append(stem)
            max_quad = max(max_quad, float(np.max(np.abs(np.asarray(ra["quad"]) - np.asarray(re["quad"])))))
            max_score = max(max_score, abs(float(ra["score"]) - float(re["score"])))
    return {"same_stems": True, "predictions_actual": count_a, "predictions_expected": count_e,
            "same_candidate_order": not different_ids, "different_images": sorted(set(different_ids)),
            "max_coordinate_abs_error": max_quad, "max_score_abs_error": max_score}


def baseline(out_path: Path | None = None, samples: dict[str, dict] | None = None):
    global _EVAL_STATE
    samples = samples or load_v28_samples()
    predictions, trace, invalid = load_v28_pool(samples)
    reference = load_cached_v28_reference()
    comparison = compare_predictions(predictions, reference)
    if not comparison.get("same_candidate_order") or comparison.get("max_coordinate_abs_error", 1) > 1e-6 or comparison.get("max_score_abs_error", 1) > 1e-7:
        raise RuntimeError(f"V28 cached baseline output mismatch: {comparison}")
    metrics, records = ev.evaluate_predictions(samples, predictions, ROOT / "results" / "pr", "A_V28")
    floor030, map030 = mh.custom_floor_metrics(samples, predictions, 0.30)
    floor045, map045 = mh.custom_floor_metrics(samples, predictions, 0.45)
    metrics["score_0.30"] = floor030
    metrics["score_0.45"] = floor045
    metrics["invalid_before_geometry"] = dict(invalid)
    metrics["v28_cache_reproduction"] = comparison
    metrics["reference_expected"] = EXPECTED_A
    checks = {
        "local_oks_ap_abs_error": abs(metrics["local_oks_ap50_95_post_polygon_nms_no_floor"] - EXPECTED_A["local_oks_ap50_95_post_polygon_nms_no_floor"]),
        "polygon_ap_abs_error": abs(metrics["polygon_ap50_95_post_polygon_nms_no_floor"] - EXPECTED_A["polygon_ap50_95_post_polygon_nms_no_floor"]),
        "floor030_exact": all(floor030[k] == v for k, v in EXPECTED_A["floor030"].items()),
        "candidate_output_match": comparison.get("same_candidate_order") and comparison.get("max_coordinate_abs_error", 1) <= 1e-6,
    }
    metrics["baseline_checks"] = checks
    if checks["local_oks_ap_abs_error"] > 1e-9 or checks["polygon_ap_abs_error"] > 1e-9 or not checks["floor030_exact"]:
        dump_json(ROOT / "logs" / "V28_BASELINE_FAILURE.json", metrics)
        raise RuntimeError(f"V28 baseline reproduction failed: {checks}")
    dump_json(ROOT / "results" / "A_V28_metrics.json", metrics)
    dump_json(ROOT / "results" / "A_V28_predictions.json", jsonable(predictions))
    _EVAL_STATE = (samples, predictions, metrics, trace)
    if out_path:
        dump_json(out_path, metrics)
    return samples, predictions, metrics, trace


def make_b_predictions(epoch: int, samples: dict[str, dict], a_predictions: dict, model_head: CornerResidualHead,
                       produce_trace=True):
    devdir = ROOT / "cache" / "dev"
    features = np.load(devdir / "features.npy", mmap_mode="r")
    maps = {name: np.load(devdir / f"{name}.npy", mmap_mode="r") for name in
            ("candidate_ids", "class_scores", "quality_scores", "boxes_original", "boxes_resized", "raw_native",
             "raw_v28", "raw_v28_online", "rois_resized", "corner_norm", "valid", "permutation",
             "fit_state", "fit_offset_resized", "image_size", "resized_size", "candidate_count")}
    rows = json.loads((devdir / "manifest_rows.json").read_text(encoding="utf-8"))
    stem_to_i = {r["stem"]: i for i, r in enumerate(rows)}
    pool = {stem: [] for stem in samples}
    candidate_records: dict[tuple[str, int], dict] = {}
    counters = Counter()
    residual_values = []
    raw_invalid = 0
    proposal_invalid = 0
    final_invalid = 0
    infer_start = time.perf_counter()
    head_device = next(model_head.parameters()).device
    model_head.eval()
    with torch.inference_mode():
        for stem in sorted(samples):
            image_i = stem_to_i[stem]
            count = int(maps["candidate_count"][image_i, 0])
            width, height = map(int, maps["image_size"][image_i, 0])
            resized_h, resized_w = map(int, maps["resized_size"][image_i, 0])
            ids = np.asarray(maps["candidate_ids"][image_i, :count], dtype=np.int32)
            if len(np.unique(ids)) != count:
                raise RuntimeError(f"duplicate dev candidate ID: {stem}")
            base_cache = V28_CACHED / f"{stem}.npz"
            with np.load(base_cache, allow_pickle=False) as ref:
                ref_ids = ref["ids"].astype(np.int32)
                if not np.array_equal(ids, ref_ids):
                    raise RuntimeError(f"dev feature IDs do not align with V28 cache: {stem}")
                base_raw = ref["raw_b"].astype(np.float64)
                class_scores = ref["p"].astype(np.float32)
                scores = ref["q"].astype(np.float32)
                boxes = ref["boxes"].astype(np.float32)
                rois = np.array(maps["rois_resized"][image_i, :count], dtype=np.float32, copy=True)
                corner_norm = roi_corner_coordinates(base_raw, rois, (width, height), (resized_h, resized_w))
                base_valid = np.zeros(count, dtype=np.uint8)
                base_canonical = []
                base_perm = []
                for j in range(count):
                    q, err, perm, valid = canonical_with_permutation(base_raw[j], width, height)
                    base_valid[j] = int(valid and err == "")
                    base_canonical.append(q)
                    base_perm.append(perm)
                raw_features = np.array(features[image_i, :count], dtype=np.float32, copy=True)
            z_parts = []
            for start in range(0, count, 64):
                end = min(count, start + 64)
                f = torch.from_numpy(raw_features[start:end]).to(head_device)
                c = torch.from_numpy(corner_norm[start:end]).to(head_device)
                z_parts.append(model_head(f, c).detach().cpu().numpy())
            z = np.concatenate(z_parts, axis=0).reshape(count, 4, 2)
            roi_t = torch.from_numpy(rois).to(head_device)
            delta = decode_residual(torch.from_numpy(z).to(head_device), roi_t, (width, height),
                                    (resized_h, resized_w)).cpu().numpy().astype(np.float64)
            proposals = base_raw + delta
            for j in range(count):
                cid = int(ids[j])
                original = base_raw[j]
                proposal = proposals[j]
                original_legal = bool(base_valid[j])
                raw_invalid += int(not original_legal)
                prop_q, prop_error, prop_perm, prop_legal = canonical_with_permutation(proposal, width, height)
                proposal_invalid += int(not (prop_legal and prop_error == ""))
                triggered = bool(np.any(delta[j] != 0.0))
                chosen, decision, reasons = guard_candidate(original, proposal, width, height, triggered)
                counters[decision] += 1
                for reason in reasons:
                    counters[f"reason:{reason}"] += 1
                final_q, final_error, final_perm, final_legal = canonical_with_permutation(chosen, width, height)
                final_valid = bool(final_legal and final_error == "")
                final_invalid += int(not final_valid)
                if not original_legal and final_valid:
                    raise RuntimeError(f"guard rescued an originally invalid candidate: {stem}/{cid}")
                if original_legal and not final_valid:
                    raise RuntimeError(f"guard produced an illegal final candidate: {stem}/{cid}")
                residual_values.extend(np.linalg.norm(delta[j], axis=1).tolist())
                rec = {
                    "stem": stem, "candidate_id": cid, "class_score": float(class_scores[j]),
                    "quality_score": float(scores[j]), "score": float(scores[j]),
                    "boxes_original": boxes[j].tolist(), "boxes_resized": maps["boxes_resized"][image_i, j].tolist(),
                    "raw_native": maps["raw_native"][image_i, j].tolist(),
                    "V28_raw": original.tolist(), "proposal_raw": proposal.tolist(), "final_raw": chosen.tolist(),
                    "V28_canonical": jsonable(base_canonical[j]), "proposal_canonical": jsonable(prop_q),
                    "final_canonical": jsonable(final_q), "V28_perm": base_perm[j], "proposal_perm": prop_perm,
                    "final_perm": final_perm, "V28_legal": original_legal,
                    "proposal_legal": bool(prop_legal and prop_error == ""), "final_legal": final_valid,
                    "decision": decision, "reasons": reasons, "delta_original": delta[j].tolist(),
                    "fit_state": maps["fit_state"][image_i, j].tolist(),
                    "fit_offset_resized": maps["fit_offset_resized"][image_i, j].tolist(),
                    "base_valid_error": "" if original_legal else "V28_original_invalid",
                    "proposal_error": prop_error, "final_error": final_error,
                    "nms_kept_A": False, "nms_kept_B": False,
                }
                candidate_records[(stem, cid)] = rec
                if final_valid:
                    pool[stem].append({"candidate_id": cid, "class_score": float(class_scores[j]),
                                       "quality_score": float(scores[j]), "score": float(scores[j]),
                                       "quad": final_q})
    predictions = prediction_from_pool(pool)
    for stem, rows in a_predictions.items():
        for row in rows:
            candidate_records[(stem, int(row["candidate_id"]))]["nms_kept_A"] = True
    for stem, rows in predictions.items():
        for row in rows:
            candidate_records[(stem, int(row["candidate_id"]))]["nms_kept_B"] = True
    elapsed = time.perf_counter() - infer_start
    trace_path = ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_candidate_trace.csv.gz"
    trace_fields = ["stem", "candidate_id", "class_score", "quality_score", "boxes_original_json", "boxes_resized_json",
                    "raw_native_json", "V28_raw_json", "proposal_raw_json", "final_raw_json",
                    "V28_canonical_json", "proposal_canonical_json", "final_canonical_json",
                    "V28_permutation", "proposal_permutation", "final_permutation",
                    "V28_legal", "proposal_legal", "final_legal", "decision", "reasons",
                    "delta_original_json", "fit_state", "fit_offset_resized_json", "NMS_A_kept",
                    "NMS_B_kept", "score_ge_030"]
    trace_rows = []
    for key in sorted(candidate_records):
        r = candidate_records[key]
        trace_rows.append({"stem": r["stem"], "candidate_id": r["candidate_id"],
                           "class_score": r["class_score"], "quality_score": r["quality_score"],
                           "boxes_original_json": r["boxes_original"], "boxes_resized_json": r["boxes_resized"],
                           "raw_native_json": r["raw_native"],
                           "V28_raw_json": r["V28_raw"], "proposal_raw_json": r["proposal_raw"],
                           "final_raw_json": r["final_raw"], "V28_canonical_json": r["V28_canonical"],
                           "proposal_canonical_json": r["proposal_canonical"], "final_canonical_json": r["final_canonical"],
                           "V28_permutation": r["V28_perm"], "proposal_permutation": r["proposal_perm"],
                           "final_permutation": r["final_perm"], "V28_legal": r["V28_legal"],
                           "proposal_legal": r["proposal_legal"], "final_legal": r["final_legal"],
                           "decision": r["decision"], "reasons": r["reasons"],
                           "delta_original_json": r["delta_original"], "fit_state": r["fit_state"],
                           "fit_offset_resized_json": r["fit_offset_resized"], "NMS_A_kept": r["nms_kept_A"],
                           "NMS_B_kept": r["nms_kept_B"], "score_ge_030": r["score"] >= 0.30})
    write_csv_gz(trace_path, trace_rows, trace_fields)
    delta = np.asarray(residual_values, dtype=np.float64)
    stats = {"epoch": epoch, "candidate_count": len(candidate_records), "guard_decisions": dict(counters),
             "raw_invalid": raw_invalid, "proposal_invalid": proposal_invalid,
             "final_invalid_before_geometry_filter": final_invalid, "final_invalid_after_filter": 0,
             "accepted_or_proposed_corner_displacement_px": finite_stats(delta),
             "head_eval_wall_seconds": elapsed, "candidate_trace": str(trace_path),
             "prediction_count_A": sum(map(len, a_predictions.values())),
             "prediction_count_B": sum(map(len, predictions.values())),
             "images_nms_changed": sum({int(r["candidate_id"]) for r in a_predictions[s]} != {int(r["candidate_id"]) for r in predictions[s]} for s in samples)}
    dump_json(ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_guard_summary.json", stats)
    dump_json(ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_B_predictions.json", jsonable(predictions))
    return predictions, candidate_records, stats


def per_gt_and_fixed_candidate(samples, a_predictions, b_predictions, candidate_records, epoch: int):
    map_a, total = ev.matches_at_floor(samples, a_predictions, 0.30)
    map_b, _ = ev.matches_at_floor(samples, b_predictions, 0.30)
    common = set(map_a) & set(map_b)
    a_only = set(map_a) - set(map_b)
    b_only = set(map_b) - set(map_a)
    rows = []
    fixed = []
    for stem, sample in samples.items():
        for gt_idx, target in enumerate(sample["targets"]):
            key = (stem, gt_idx)
            ra, rb = map_a.get(key), map_b.get(key)
            status = "common" if ra and rb else "A_only" if ra else "B_only" if rb else "neither"
            rows.append({"stem": stem, "gt_index": gt_idx, "status_floor030": status,
                         "gt_quad": json.dumps(target, separators=(",", ":")),
                         "A_candidate_id": "" if not ra else int(ra["candidate_id"]),
                         "B_candidate_id": "" if not rb else int(rb["candidate_id"]),
                         "A_iou": "" if not ra else float(ra["similarity"]),
                         "B_iou": "" if not rb else float(rb["similarity"]),
                         "A_quad": "" if not ra else json.dumps(ra["quad"], separators=(",", ":")),
                         "B_quad": "" if not rb else json.dumps(rb["quad"], separators=(",", ":"))})
            # Fixed A association: same candidate ID and GT are evaluated in B,
            # regardless of whether B's independent matching selects it.
            if ra:
                cid = int(ra["candidate_id"])
                candidate = candidate_records[(stem, cid)]
                bq = candidate["final_canonical"]
                if bq is None:
                    continue
                bq = np.asarray(bq, dtype=np.float64)
                aq = np.asarray(ra["quad"], dtype=np.float64)
                t = np.asarray(target, dtype=np.float64)
                scale = max(math.sqrt(geom.polygon_area(target)), 1.0)
                ea = np.linalg.norm(aq - t, axis=1) / scale
                eb = np.linalg.norm(bq - t, axis=1) / scale
                d = float(eb.mean() - ea.mean())
                fixed.append({"stem": stem, "gt_index": gt_idx, "candidate_id": cid,
                              "A_error_normalized_mean": float(ea.mean()),
                              "B_error_normalized_mean": float(eb.mean()),
                              "delta_B_minus_A": d, "A_error_px_mean": float(np.linalg.norm(aq-t, axis=1).mean()),
                              "B_error_px_mean": float(np.linalg.norm(bq-t, axis=1).mean()),
                              "A_quad": aq.tolist(), "B_quad": bq.tolist(), "GT_quad": t.tolist(),
                              "small": bool(sample["small"][gt_idx]), "tiny": bool(sample["tiny"][gt_idx]),
                              "boundary": bool(sample["boundary"][gt_idx]), "candidate_still_final_B": candidate["nms_kept_B"]})
    write_csv_gz(ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_gt_attribution.csv.gz", rows)
    write_csv_gz(ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_fixed_A_geometry.csv.gz", fixed)
    fixed_delta = np.asarray([r["delta_B_minus_A"] for r in fixed], dtype=np.float64)
    fixed_stats = {"A_matched_gt_count": len(map_a), "paired_fixed_candidate_count": len(fixed),
                   "all_A_match_keys_count": len(map_a), "fixed_candidate_geometry_mean_delta": float(fixed_delta.mean()) if len(fixed_delta) else None,
                   "improved": int(np.sum(fixed_delta < -1e-12)), "tied": int(np.sum(np.abs(fixed_delta) <= 1e-12)),
                   "worsened": int(np.sum(fixed_delta > 1e-12)),
                   "delta_distribution": finite_stats(fixed_delta),
                   "common_floor030_gt": len(common), "A_only_floor030_gt": len(a_only),
                   "B_only_floor030_gt": len(b_only), "neither_floor030_gt": total-len(map_a|map_b)}
    nms_rows = []
    for stem in samples:
        a_ids = {int(r["candidate_id"]) for r in a_predictions[stem]}
        b_ids = {int(r["candidate_id"]) for r in b_predictions[stem]}
        if a_ids != b_ids:
            nms_rows.append({"stem": stem, "A_only_candidate_ids": sorted(a_ids-b_ids),
                             "B_only_candidate_ids": sorted(b_ids-a_ids),
                             "A_output_count": len(a_ids), "B_output_count": len(b_ids)})
    write_csv_gz(ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_nms_changes.csv.gz", nms_rows)
    return {"floor030_gt_status": fixed_stats, "fixed_A_geometry": fixed_stats,
            "nms_changed_images": len(nms_rows), "nms_changed_rows": nms_rows}


def make_figures(samples, fixed_rows, gt_rows, candidate_records, epoch: int):
    fig_dir = ROOT / "figures" / f"epoch_{epoch:02d}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(fixed_rows, key=lambda r: r["delta_B_minus_A"])
    chosen = []
    for r in candidates[:5]:
        chosen.append(("largest_fixed_A_improvement", r))
    for r in candidates[-5:][::-1]:
        chosen.append(("largest_fixed_A_regression", r))
    status_rows = [r for r in gt_rows if r["status_floor030"] in ("A_only", "B_only")]
    for status in ("A_only", "B_only"):
        for r in status_rows:
            if r["status_floor030"] == status:
                match = next((x for x in fixed_rows if x["stem"] == r["stem"] and x["gt_index"] == int(r["gt_index"])), None)
                if match is not None:
                    chosen.append((f"{status}_fixed_A_candidate", match))
                    break
    out = []
    seen = set()
    for label, row in chosen:
        key = (row["stem"], row["gt_index"], label)
        if key in seen:
            continue
        seen.add(key)
        source = next(x for x in json.loads((ROOT / "cache" / "dev" / "manifest_rows.json").read_text(encoding="utf-8")) if x["stem"] == row["stem"])
        with Image.open(source["image"]) as im:
            image = im.convert("RGB")
        draw = ImageDraw.Draw(image)
        colors = {"A": (255, 215, 0), "B": (0, 240, 100), "GT": (255, 50, 50)}
        for tag, quad in (("A", row["A_quad"]), ("B", row["B_quad"]), ("GT", row["GT_quad"])):
            pts = [(float(x), float(y)) for x, y in quad]
            draw.line(pts + [pts[0]], fill=colors[tag], width=3)
            for idx, pt in enumerate(pts):
                draw.ellipse((pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4), fill=colors[tag])
                draw.text((pt[0]+4, pt[1]+3), f"{tag}{idx}", fill=colors[tag])
        filename = f"{len(out):02d}_{label}_{row['stem']}_gt{row['gt_index']}.png"
        path = fig_dir / filename
        image.save(path)
        out.append({"label": label, "stem": row["stem"], "gt_index": row["gt_index"],
                    "delta_norm": row["delta_B_minus_A"], "file": filename})
    html = ["<!doctype html><meta charset='utf-8'><title>V31 epoch figures</title>",
            "<style>body{font:14px sans-serif;background:#111;color:#eee}figure{display:inline-block;vertical-align:top;width:48%}img{max-width:100%}figcaption{padding:6px}</style>"]
    for row in out:
        html.append(f"<figure><img src='{row['file']}'><figcaption>{row['label']} | {row['stem']} GT {row['gt_index']} | Δ normalized error {row['delta_norm']:.6f}; yellow=A, green=B, red=GT</figcaption></figure>")
    (fig_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    return out


def promotion_gate(metrics_a, metrics_b, floors_a, floors_b, common_stats, bootstrap, illegal_a, illegal_b):
    ap_local_delta = metrics_b["local_oks_ap50_95_post_polygon_nms_no_floor"] - metrics_a["local_oks_ap50_95_post_polygon_nms_no_floor"]
    ap_poly_delta = metrics_b["polygon_ap50_95_post_polygon_nms_no_floor"] - metrics_a["polygon_ap50_95_post_polygon_nms_no_floor"]
    ci = {r["metric"]: r for r in bootstrap}
    tp_budgets = all(metrics_b["operating_points"][f"fp_budget_{b}"]["tp"] >= metrics_a["operating_points"][f"fp_budget_{b}"]["tp"] for b in (100, 200))
    groups = all(floors_b["0.30"][f"{k}_detected"] >= floors_a["0.30"][f"{k}_detected"] for k in ("small", "tiny", "boundary"))
    paired_error = common_stats["paired_common_gt_normalized"]
    ap_ci_gain = ((ap_local_delta >= 0.002 and ci["local_oks"]["ci95_low"] > 0) or
                  (ap_poly_delta >= 0.002 and ci["polygon"]["ci95_low"] > 0))
    tests = {
        "both_AP_non_decrease": ap_local_delta >= 0 and ap_poly_delta >= 0,
        "one_AP_gain_at_least_0.002_and_CI_lower_positive": ap_ci_gain,
        "TP_at_FP100_and_FP200_non_decrease": tp_budgets,
        "floor030_small_tiny_boundary_non_decrease": groups,
        "common_GT_normalized_mean_non_increase": paired_error["B_mean"] <= paired_error["A_mean"],
        "common_GT_normalized_P95_non_increase": paired_error["B_p95"] <= paired_error["A_p95"],
        "final_illegal_non_increase": illegal_b <= illegal_a,
    }
    return {"tests": tests, "all_pass": all(tests.values()), "AP_delta": {"local_oks": ap_local_delta, "polygon": ap_poly_delta},
            "bootstrap_CI": {m: {"low": ci[m]["ci95_low"], "high": ci[m]["ci95_high"]} for m in ci}}


def evaluate_epoch(epoch: int, checkpoint_path: Path | None = None, do_bootstrap=True, do_online_if_pass=True):
    global _EVAL_STATE
    if _EVAL_STATE is None:
        metrics_path = ROOT / "results" / "A_V28_metrics.json"
        predictions_path = ROOT / "results" / "A_V28_predictions.json"
        if metrics_path.exists() and predictions_path.exists():
            _EVAL_STATE = (load_v28_samples(), json.loads(predictions_path.read_text(encoding="utf-8")),
                           json.loads(metrics_path.read_text(encoding="utf-8")), {})
        else:
            baseline()
    samples, a_predictions, metrics_a, a_trace = _EVAL_STATE
    if checkpoint_path is None:
        checkpoint_path = ROOT / "checkpoints" / f"epoch_{epoch:02d}.pt"
    ckpt = safe_torch_load(checkpoint_path, "cpu")
    head = CornerResidualHead().cuda().eval()
    head.load_state_dict(ckpt["residual_head"], strict=True)
    head.requires_grad_(False)
    t0 = time.perf_counter()
    b_predictions, candidate_records, guard_stats = make_b_predictions(epoch, samples, a_predictions, head)
    metrics_b, _records = ev.evaluate_predictions(samples, b_predictions, ROOT / "results" / "pr", f"B_epoch{epoch:02d}")
    floors_b = {}
    for floor in (0.30, 0.45):
        summary, _ = mh.custom_floor_metrics(samples, b_predictions, floor)
        floors_b[f"{floor:.2f}"] = summary
    floors_a = {}
    for floor in (0.30, 0.45):
        summary, _ = mh.custom_floor_metrics(samples, a_predictions, floor)
        floors_a[f"{floor:.2f}"] = summary
    metrics_b["score_0.30"] = floors_b["0.30"]
    metrics_b["score_0.45"] = floors_b["0.45"]
    attribution = per_gt_and_fixed_candidate(samples, a_predictions, b_predictions, candidate_records, epoch)
    # Common-TP geometry is measured on the exact same GT keys in A and B.
    map_a, _ = ev.matches_at_floor(samples, a_predictions, 0.30)
    map_b, _ = ev.matches_at_floor(samples, b_predictions, 0.30)
    common = sorted(set(map_a) & set(map_b))
    errors_a, errors_b = [], []
    for stem, gt_idx in common:
        target = np.asarray(samples[stem]["targets"][gt_idx], dtype=np.float64)
        area_scale = max(math.sqrt(geom.polygon_area(target.tolist())), 1.0)
        for mapping, collection in ((map_a, errors_a), (map_b, errors_b)):
            pred = np.asarray(mapping[(stem, gt_idx)]["quad"], dtype=np.float64)
            collection.extend((np.linalg.norm(pred - target, axis=1) / area_scale).tolist())
    common_stats = {"matched_A": len(map_a), "matched_B": len(map_b), "common_GT": len(common),
                    "A_only": len(set(map_a) - set(map_b)), "B_only": len(set(map_b) - set(map_a)),
                    "paired_common_gt_normalized": {"A_mean": float(np.mean(errors_a)) if errors_a else None,
                                                     "B_mean": float(np.mean(errors_b)) if errors_b else None,
                                                     "A_p95": float(np.percentile(errors_a, 95)) if errors_a else None,
                                                     "B_p95": float(np.percentile(errors_b, 95)) if errors_b else None}}
    metrics_b["matched_GT_error_common_only"] = common_stats
    metrics_b["candidate_geometry_and_guard"] = guard_stats
    metrics_b["fixed_A_GT_association"] = attribution["fixed_A_geometry"]
    metrics_b["NMS_selection_changes"] = {k: attribution[k] for k in ("nms_changed_images",)}
    metrics_b["elapsed_cached_evaluation_seconds"] = time.perf_counter() - t0
    metrics_b["local_oks_is_approximation_not_official_metric"] = True
    illegal_a = int(guard_stats["raw_invalid"])
    illegal_b = int(guard_stats["final_invalid_before_geometry_filter"])
    bootstrap_summary = []
    gate = None
    if epoch == 3 and do_bootstrap:
        draws, bootstrap_summary = mh.paired_bootstrap(samples, {"A": a_predictions, "B": b_predictions},
                                                       seed=20261031, replicates=1000)
        write_csv_gz(ROOT / "results" / "bootstrap_epoch03_draws.csv.gz", draws)
        dump_json(ROOT / "results" / "bootstrap_epoch03_summary.json", bootstrap_summary)
        gate = promotion_gate(metrics_a, metrics_b, floors_a, floors_b, common_stats,
                              bootstrap_summary, illegal_a, illegal_b)
        dump_json(ROOT / "results" / "promotion_gate.json", gate)
        if gate["all_pass"] and do_online_if_pass:
            online = online_confirmation(ckpt, samples)
            metrics_b["online_confirmation"] = online
    if bootstrap_summary:
        metrics_b["paired_bootstrap"] = bootstrap_summary
    if gate is not None:
        metrics_b["promotion_gate"] = gate
    dump_json(ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_metrics.json", metrics_b)
    figures = make_figures(samples, list_from_fixed_csv(epoch), read_gt_rows(epoch), candidate_records, epoch) if epoch == 3 else []
    dump_json(ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_evaluation_status.json",
              {"epoch": epoch, "complete": True, "metrics": str((ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_metrics.json").resolve()),
               "figures": figures, "promotion_gate": gate})
    print(json.dumps({"stage": "dev_evaluation", "epoch": epoch,
                      "A_local": metrics_a["local_oks_ap50_95_post_polygon_nms_no_floor"],
                      "B_local": metrics_b["local_oks_ap50_95_post_polygon_nms_no_floor"],
                      "A_polygon": metrics_a["polygon_ap50_95_post_polygon_nms_no_floor"],
                      "B_polygon": metrics_b["polygon_ap50_95_post_polygon_nms_no_floor"],
                      "floor030": floors_b["0.30"], "promotion_gate": gate}, ensure_ascii=False), flush=True)
    return metrics_b


def list_from_fixed_csv(epoch: int):
    path = ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_fixed_A_geometry.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("A_quad", "B_quad", "GT_quad"):
            row[key] = json.loads(row[key])
        row["delta_B_minus_A"] = float(row["delta_B_minus_A"])
        row["gt_index"] = int(row["gt_index"])
        for key in ("small", "tiny", "boundary", "candidate_still_final_B"):
            row[key] = row[key] == "True"
    return rows


def read_gt_rows(epoch: int):
    path = ROOT / "results" / "epochs" / f"epoch_{epoch:02d}_gt_attribution.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def online_confirmation(ckpt, samples=None, suffix=""):
    """Fresh full dev forward; if samples is None, lock predictions before loading GT."""
    from v31_core import install_capture
    model, quality_head, mean, std = load_models()
    cap = install_capture(model)
    head = CornerResidualHead().cuda().eval()
    head.load_state_dict(ckpt["residual_head"], strict=True)
    rows = rank.read_rows(MANIFEST, "dev")
    pools = {"A": {r["stem"]: [] for r in rows}, "B": {r["stem"]: [] for r in rows}}
    deltas = []
    active_gpu_ms = 0.0
    nonidentity = 0
    checks = []
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            out = run_image(model, cap, quality_head, mean, std, row["image"])
            active_gpu_ms += out["model_gpu_ms"] + out["decoder_gpu_ms"] + out["quality_and_spatial_gpu_ms"]
            width, height = out["original_size"]
            resized_h, resized_w = out["resized_size"]
            feat = torch.from_numpy(out["spatial_features"]).cuda()
            coords = torch.from_numpy(out["roi_corner_norm"]).cuda()
            z = head(feat, coords).reshape(-1, 4, 2)
            roi = torch.from_numpy(out["rois_resized"]).cuda()
            delta = decode_residual(z, roi, out["original_size"], out["resized_size"]).cpu().numpy().astype(np.float64)
            proposal = out["raw_v28"] + delta
            for j, cid in enumerate(out["candidate_ids"]):
                qA, errA = rank.parse_prediction(out["raw_v28"][j].tolist(), width, height)
                proposal_q, proposal_err, _proposal_perm, proposal_valid = canonical_with_permutation(proposal[j], width, height)
                chosen, decision, _reasons = guard_candidate(out["raw_v28"][j], proposal[j], width, height,
                                                             bool(np.any(delta[j] != 0.0)))
                qB, errB = rank.parse_prediction(chosen.tolist(), width, height)
                if qA is not None and not errA:
                    pools["A"][row["stem"]].append({"candidate_id": int(cid), "score": float(out["quality_scores"][j]),
                                                      "quality_score": float(out["quality_scores"][j]), "class_score": float(out["class_scores"][j]), "quad": qA})
                if qB is not None and not errB:
                    pools["B"][row["stem"]].append({"candidate_id": int(cid), "score": float(out["quality_scores"][j]),
                                                      "quality_score": float(out["quality_scores"][j]), "class_score": float(out["class_scores"][j]), "quad": qB})
                nonidentity += int(decision == "accepted")
                deltas.extend(np.linalg.norm(delta[j], axis=1).tolist())
            if i % 50 == 0 or i == len(rows):
                print(json.dumps({"stage": "online_confirmation_forward", "images": f"{i}/{len(rows)}"}), flush=True)
    predictions = {arm: prediction_from_pool(pools[arm]) for arm in ("A", "B")}
    prediction_path = ROOT / "results" / f"online_confirmation_predictions{suffix}.json"
    dump_json(prediction_path, jsonable(predictions))
    locked_sha256 = sha256(prediction_path)
    lock_path = ROOT / "results" / f"online_confirmation_predictions{suffix}.lock.json"
    dump_json(lock_path, {"predictions_path": str(prediction_path.resolve()),
                          "predictions_sha256": locked_sha256, "image_count": len(rows),
                          "GT_loaded_before_lock": False,
                          "prediction_output_arms": ["A", "B"]})
    loaded_after_lock = samples is None
    if samples is None:
        # This branch is used by the independent `--stage online` process.
        # No annotations are opened until the image-only predictions and their
        # checksum lock have been written above.
        samples = ev.load_dev(MANIFEST, "dev")
    online_metrics = {}
    for arm in ("A", "B"):
        met, _ = ev.evaluate_predictions(samples, predictions[arm], ROOT / "results" / "pr", f"online_{arm}{suffix}")
        met["score_0.30"] = mh.custom_floor_metrics(samples, predictions[arm], 0.30)[0]
        met["score_0.45"] = mh.custom_floor_metrics(samples, predictions[arm], 0.45)[0]
        online_metrics[arm] = met
    record = {"status": "complete", "images": len(rows), "fresh_image_only_forward": True,
              "GT_loaded_only_after_predictions_locked": loaded_after_lock,
              "predictions_path": str(prediction_path.resolve()), "predictions_sha256": locked_sha256,
              "prediction_lock_path": str(lock_path.resolve()), "GPU_active_seconds": active_gpu_ms / 1000.0,
              "accepted_candidates": nonidentity, "corner_delta_px": finite_stats(np.asarray(deltas)),
              "metrics": online_metrics, "scores_online_path": "V10 quality head from original uncorrected 1083D feature, float16 roundtrip, frozen mean/std"}
    dump_json(ROOT / "results" / f"online_confirmation_metrics{suffix}.json", record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("baseline", "epoch", "online"), required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.stage == "baseline":
        baseline()
    elif args.stage == "online":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --stage online")
        ckpt = safe_torch_load(args.checkpoint, "cpu")
        if (ckpt.get("run_config_sha256") != sha256(ROOT / "RUN_CONFIG.json")
                or ckpt.get("detector_sha256") != BASE_SHA
                or ckpt.get("quality_head_sha256") != HEAD_SHA):
            raise RuntimeError("online checkpoint does not match frozen V31 configuration or weights")
        online_confirmation(ckpt, samples=None, suffix="_staged")
    else:
        if args.epoch not in (1, 2, 3):
            raise ValueError("epoch must be 1, 2 or 3")
        evaluate_epoch(args.epoch, args.checkpoint)


if __name__ == "__main__":
    main()
