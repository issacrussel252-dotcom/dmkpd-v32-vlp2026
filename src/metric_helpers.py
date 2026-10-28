import math,csv,gzip
from pathlib import Path
from collections import defaultdict
import os
import numpy as np
import v9_evaluate as ev
def custom_floor_metrics(samples, predictions, floor: float):
    records, total = ev.ranked_matches(samples, predictions, ev.quad_iou, floor)
    tp = sum(int(r["tp"]) for r in records)
    fp = len(records) - tp
    mapping = {(r["stem"], r["gt_index"]): r for r in records if r["tp"]}
    summary = ev.corner_summary(samples, mapping)
    n = sum(len(rows) for rows in predictions.values() if rows)
    predictions_over_floor = sum(float(row["score"]) >= floor for rows in predictions.values() for row in rows)
    # Corner error tails use the same one-to-one evaluator matches.
    errors, norm = [], []
    for (stem, gt_idx), record in mapping.items():
        target = samples[stem]["targets"][gt_idx]
        pred = record["quad"]
        area_scale = max(math.sqrt(ev.polygon_area(target)), 1.0)
        for i in range(4):
            e = math.dist(pred[i], target[i])
            errors.append(e)
            norm.append(e / area_scale)
    summary.update({
        "tp": tp, "fp": fp, "fn": total - tp,
        "predictions_after_floor": predictions_over_floor,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(total, 1),
        "corner_error_p90_px": float(np.percentile(errors, 90)) if errors else None,
        "corner_error_p95_px": float(np.percentile(errors, 95)) if errors else None,
        "corner_error_p99_px": float(np.percentile(errors, 99)) if errors else None,
        "corner_error_normalized_p95": float(np.percentile(norm, 95)) if norm else None,
        "corner_error_normalized_p99": float(np.percentile(norm, 99)) if norm else None,
    })
    return summary, mapping

def gt_comparison(samples, maps, out_path: Path):
    all_keys = {(stem, i) for stem, sample in samples.items() for i in range(len(sample["targets"]))}
    rows = []
    for stem, gt_idx in sorted(all_keys):
        target = samples[stem]["targets"][gt_idx]
        a = maps["A"].get((stem, gt_idx)); b = maps["B"].get((stem, gt_idx))
        status = "common" if a and b else ("B_only" if b else ("A_only" if a else "neither"))
        def fields(rec):
            if not rec:
                return {"candidate_id": "", "score": "", "polygon_iou": "", "quad": "", "corner_error_mean_px": "", "corner_error_mean_normalized": ""}
            q = rec["quad"]
            err = [math.dist(q[i], target[i]) for i in range(4)]
            scale = max(math.sqrt(ev.polygon_area(target)), 1.0)
            return {"candidate_id": rec["candidate_id"], "score": rec["score"], "polygon_iou": ev.quad_iou(q, target), "quad": json.dumps(q, separators=(",", ":")), "corner_error_mean_px": float(np.mean(err)), "corner_error_mean_normalized": float(np.mean(err) / scale)}
        row = {"stem": stem, "gt_index": gt_idx, "status_at_floor_030": status, "polygon_area": ev.polygon_area(target), "gt_quad": json.dumps(target, separators=(",", ":"))}
        row.update({f"A_{k}": v for k, v in fields(a).items()})
        row.update({f"B_{k}": v for k, v in fields(b).items()})
        rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    return rows

def paired_bootstrap(samples, predictions, seed=20261027, replicates=1000):
    stems = sorted(samples)
    rng = np.random.default_rng(seed)
    thresholds = [round(.5 + .05*i, 2) for i in range(10)]
    similarities = {"polygon": ev.quad_iou, "local_oks": ev.local_oks}
    # Precompute per-image TP identities; matching is image-local and uses the
    # exact frozen score/candidate ordering, so each repeated copy has the same
    # within-image matches before the global AP rank is recomputed.
    per_arm = {}
    for arm in ("A", "B"):
        per_arm[arm] = {}
        for metric, similarity in similarities.items():
            per_arm[arm][metric] = []
            for threshold in thresholds:
                group = []
                for stem in stems:
                    rows = sorted(predictions[arm][stem], key=lambda x: (-float(x["score"]), int(x["candidate_id"])))
                    used = set();flags = []
                    for row in rows:
                        choices = [(similarity(row["quad"], gt), gi) for gi, gt in enumerate(samples[stem]["targets"]) if gi not in used]
                        sim, gt_idx = max(choices, key=lambda z: (z[0], -z[1]), default=(0.0, -1))
                        hit = sim >= threshold
                        flags.append(int(hit))
                        if hit: used.add(gt_idx)
                    group.append({"scores": np.asarray([float(x["score"]) for x in rows], dtype=np.float64), "cids": np.asarray([int(x["candidate_id"]) for x in rows], dtype=np.int32), "tp": np.asarray(flags, dtype=np.int8)})
                per_arm[arm][metric].append(group)
    recalls = np.linspace(0., 1., 101)
    def ap_from_arrays(scores, copy_ids, cids, flags, total_gt):
        if total_gt <= 0 or not len(scores): return 0.0
        order = np.lexsort((cids, copy_ids, -scores))
        tp = flags[order].astype(np.int64)
        cum_tp = np.cumsum(tp)
        cum_fp = np.arange(1, len(tp)+1, dtype=np.int64) - cum_tp
        recall = cum_tp / total_gt
        precision = cum_tp / np.maximum(cum_tp+cum_fp, 1)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        idx = np.searchsorted(recall, recalls, side="left")
        sampled = np.zeros_like(recalls)
        mask = idx < len(envelope)
        sampled[mask] = envelope[idx[mask]]
        return float(sampled.mean())
    draws = []
    gt_counts = np.asarray([len(samples[s]["targets"]) for s in stems], dtype=np.int32)
    for rep in range(replicates):
        draw = rng.integers(0, len(stems), size=len(stems))
        total_gt = int(gt_counts[draw].sum())
        record = {"replicate": rep, "drawn_images": len(draw), "ground_truth_count": total_gt,
                  "sampled_indices": draw.tolist()}
        duplicate_ordinals = defaultdict(int)
        copy_names = []
        for sampled_idx in draw:
            source_stem = stems[int(sampled_idx)]
            duplicate_ordinals[source_stem] += 1
            # The unique bootstrap image ID is also the evaluator's lexical
            # global-score tie key, followed by the original candidate ID.
            copy_names.append(f"{source_stem}__copy{duplicate_ordinals[source_stem]:04d}")
        for arm in ("A", "B"):
            for metric in similarities:
                level_aps = []
                for li in range(len(thresholds)):
                    chunks = [per_arm[arm][metric][li][idx] for idx in draw]
                    lengths = [len(x["scores"]) for x in chunks]
                    scores = np.concatenate([x["scores"] for x in chunks]) if sum(lengths) else np.empty(0,dtype=np.float64)
                    cids = np.concatenate([x["cids"] for x in chunks]) if sum(lengths) else np.empty(0,dtype=np.int32)
                    flags = np.concatenate([x["tp"] for x in chunks]) if sum(lengths) else np.empty(0,dtype=np.int8)
                    copy_ids = np.concatenate([np.repeat(copy_names[draw_i], length) for draw_i, length in enumerate(lengths)]) if sum(lengths) else np.empty(0,dtype="U16")
                    level_aps.append(ap_from_arrays(scores, copy_ids, cids, flags, total_gt))
                record[f"{arm}_{metric}_AP50_95"] = float(np.mean(level_aps))
                for level, value in zip(thresholds, level_aps): record[f"{arm}_{metric}_AP{int(level*100):02d}"] = value
        record["delta_local_oks_AP50_95"] = record["B_local_oks_AP50_95"] - record["A_local_oks_AP50_95"]
        record["delta_polygon_AP50_95"] = record["B_polygon_AP50_95"] - record["A_polygon_AP50_95"]
        draws.append(record)
        if (rep + 1) % 100 == 0:
            print(f"bootstrap {rep+1}/{replicates}", flush=True)
    observed = {}
    for arm in ("A", "B"):
        observed[arm] = {}
        for metric in similarities:
            levels = []
            for li in range(len(thresholds)):
                chunks = [per_arm[arm][metric][li][idx] for idx in range(len(stems))]
                lengths = [len(x["scores"]) for x in chunks]
                scores = np.concatenate([x["scores"] for x in chunks]) if sum(lengths) else np.empty(0,dtype=np.float64)
                cids = np.concatenate([x["cids"] for x in chunks]) if sum(lengths) else np.empty(0,dtype=np.int32)
                flags = np.concatenate([x["tp"] for x in chunks]) if sum(lengths) else np.empty(0,dtype=np.int8)
                copy_ids = np.concatenate([np.repeat(stems[i], length) for i,length in enumerate(lengths)]) if sum(lengths) else np.empty(0,dtype="U16")
                levels.append(ap_from_arrays(scores, copy_ids, cids, flags, int(gt_counts.sum())))
            observed[arm][metric] = float(np.mean(levels))
    summary = []
    for metric in ("local_oks", "polygon"):
        deltas = np.asarray([r[f"delta_{metric}_AP50_95"] for r in draws])
        summary.append({"metric": metric, "replicates": replicates, "seed": seed, "observed_full_dev_B_minus_A": observed["B"][metric]-observed["A"][metric], "ci95_low": float(np.percentile(deltas, 2.5)), "ci95_high": float(np.percentile(deltas,97.5)), "bootstrap_draw_mean": float(deltas.mean()), "bootstrap_unit": "image; manifest source_group singleton per dev image"})
    return draws, summary
