"""V7-local geometry and LabelMe parsing helpers.

This module intentionally has no imports from historical experiment folders.
The ordering rule is applied to both training targets and evaluation outputs.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import os
import numpy as np


def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def canonical_quad(points: Iterable[Iterable[float]]) -> list[tuple[float, float]]:
    """Return one deterministic cyclic ordering for a convex quadrilateral."""
    values = [(float(point[0]), float(point[1])) for point in points]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in values):
        raise ValueError("quad contains non-finite coordinates")
    unique: list[tuple[float, float]] = []
    for point in values:
        if not any(math.hypot(point[0] - other[0], point[1] - other[1]) <= 1e-6 for other in unique):
            unique.append(point)
    if len(unique) != 4:
        raise ValueError(f"expected four unique points, got {len(unique)}")
    polygon = hull(unique)
    if len(polygon) != 4:
        raise ValueError("points do not form a convex quadrilateral")
    start = min(range(4), key=lambda index: (polygon[index][0] + polygon[index][1], polygon[index][1], polygon[index][0]))
    forward = [polygon[(start + index) % 4] for index in range(4)]
    reverse = [polygon[(start - index) % 4] for index in range(4)]
    forward_key = (forward[1][0] - forward[3][0], forward[3][1] - forward[1][1])
    reverse_key = (reverse[1][0] - reverse[3][0], reverse[3][1] - reverse[1][1])
    return forward if forward_key >= reverse_key else reverse


def signed_area(points: Iterable[tuple[float, float]]) -> float:
    values = list(points)
    return sum(values[index][0] * values[(index + 1) % len(values)][1] - values[(index + 1) % len(values)][0] * values[index][1] for index in range(len(values))) / 2.0


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    values = list(points)
    return abs(signed_area(values)) if len(values) >= 3 else 0.0


def ordered_quad_is_valid(points: list[tuple[float, float]] | None) -> bool:
    if points is None:
        return False
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        return False
    if any(math.dist(tuple(values[index]), tuple(values[(index + 1) % 4])) <= 1e-6 for index in range(4)):
        return False
    crosses = np.asarray([cross(tuple(values[index]), tuple(values[(index + 1) % 4]), tuple(values[(index + 2) % 4])) for index in range(4)])
    return bool(np.all(crosses > 1e-9) or np.all(crosses < -1e-9))


def convex_intersection(subject: list[tuple[float, float]], clip: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(subject) < 3 or len(clip) < 3:
        return []
    orientation = 1.0 if signed_area(clip) >= 0 else -1.0

    def inside(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> bool:
        return orientation * cross(start, end, point) >= -1e-7

    def intersect(p1: tuple[float, float], p2: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        denominator = (p1[0] - p2[0]) * (a[1] - b[1]) - (p1[1] - p2[1]) * (a[0] - b[0])
        if abs(denominator) < 1e-12:
            return p2
        t = ((p1[0] - a[0]) * (a[1] - b[1]) - (p1[1] - a[1]) * (a[0] - b[0])) / denominator
        return p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])

    output = subject
    for index, start in enumerate(clip):
        end = clip[(index + 1) % len(clip)]
        previous_points = output
        output = []
        if not previous_points:
            break
        previous = previous_points[-1]
        for current in previous_points:
            if inside(current, start, end):
                if not inside(previous, start, end):
                    output.append(intersect(previous, current, start, end))
                output.append(current)
            elif inside(previous, start, end):
                output.append(intersect(previous, current, start, end))
            previous = current
    return output


def quad_iou(first: list[tuple[float, float]] | None, second: list[tuple[float, float]] | None) -> float:
    if first is None or second is None:
        return 0.0
    try:
        a = canonical_quad(first)
        b = canonical_quad(second)
    except (TypeError, ValueError, IndexError):
        return 0.0
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    intersection = polygon_area(convex_intersection(a, b))
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 1e-12 else 0.0


def local_oks(prediction: list[tuple[float, float]], target: list[tuple[float, float]], sigma: float = 0.1) -> float:
    try:
        pred = canonical_quad(prediction)
        gt = canonical_quad(target)
    except (TypeError, ValueError, IndexError):
        return 0.0
    area = max(polygon_area(gt), 1.0)
    scale = math.sqrt(area)
    denominator = 2.0 * (scale * sigma) ** 2
    values = [math.exp(-((px - tx) ** 2 + (py - ty) ** 2) / denominator) for (px, py), (tx, ty) in zip(pred, gt)]
    return float(np.mean(values))


def parse_shapes(path: Path, stem: str) -> list[list[tuple[float, float]]]:
    """Parse only DPM LabelMe shapes using the frozen V7 data rule."""
    data = json.loads(path.read_text(encoding="utf-8"))
    result: list[list[tuple[float, float]]] = []
    for shape_index, shape in enumerate(data.get("shapes", [])):
        if str(shape.get("label", "")).casefold() != "dpm":
            continue
        raw = [(float(point[0]), float(point[1])) for point in shape.get("points", [])]
        if len(raw) == 5:
            for index in range(len(raw)):
                if any(math.hypot(raw[index][0] - raw[other][0], raw[index][1] - raw[other][1]) <= 1e-6 for other in range(len(raw)) if other != index):
                    raw.pop(index)
                    break
        # This is the previously audited source-data repair, kept local to V7.
        if stem == "005308" and shape_index == 2:
            raw = [(x, max(0.0, y)) for x, y in raw]
        result.append(canonical_quad(raw))
    if not result:
        raise ValueError(f"no DPM shapes in {path}")
    return result


def greedy_matches(predictions: list[dict[str, Any]], targets: list[list[tuple[float, float]]], similarity: Callable[[Any, Any], float], threshold: float) -> list[tuple[int, int, float]]:
    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    order = sorted(range(len(predictions)), key=lambda index: (-float(predictions[index]["score"]), int(predictions[index].get("candidate_id", index))))
    for prediction_index in order:
        choices = [(float(similarity(predictions[prediction_index]["quad"], target)), target_index) for target_index, target in enumerate(targets) if target_index not in used]
        if not choices:
            continue
        score, target_index = max(choices, key=lambda item: (item[0], -item[1]))
        if score >= threshold:
            used.add(target_index)
            matches.append((prediction_index, target_index, score))
    return matches


def average_precision(samples: dict[str, dict[str, Any]], similarity: Callable[[Any, Any], float], level: float, score_floor: float | None = None) -> float:
    ranked: list[tuple[float, str, int, list[tuple[float, float]]]] = []
    total_targets = 0
    for stem, sample in samples.items():
        total_targets += len(sample["targets"])
        for row in sample["predictions"]:
            if score_floor is not None and float(row["score"]) < score_floor:
                continue
            ranked.append((float(row["score"]), stem, int(row.get("candidate_id", 0)), row["quad"]))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    used: dict[str, set[int]] = defaultdict(set)
    tp: list[int] = []
    fp: list[int] = []
    for _, stem, _, quad in ranked:
        choices = [(float(similarity(quad, target)), index) for index, target in enumerate(samples[stem]["targets"]) if index not in used[stem]]
        best, target_index = max(choices, default=(0.0, -1))
        is_tp = int(best >= level)
        if is_tp:
            used[stem].add(target_index)
        tp.append(is_tp)
        fp.append(1 - is_tp)
    if not tp or total_targets == 0:
        return 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recalls = cumulative_tp / total_targets
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    envelope = [float(np.max(precisions[recalls >= recall])) if np.any(recalls >= recall) else 0.0 for recall in np.linspace(0.0, 1.0, 101)]
    return float(np.mean(envelope))


def fixed_metrics(samples: dict[str, dict[str, Any]], score_floor: float | None, similarity: Callable[[Any, Any], float] = quad_iou) -> dict[str, Any]:
    target_count = sum(len(sample["targets"]) for sample in samples.values())
    predictions_count = 0
    tp = 0
    small_total = small_tp = 0
    tiny_total = tiny_tp = 0
    matched_errors: list[float] = []
    matched_normalized_errors: list[float] = []
    invalid = 0
    for sample in samples.values():
        visible = [row for row in sample["predictions"] if score_floor is None or float(row["score"]) >= score_floor]
        predictions_count += len(visible)
        invalid += sum(int(not ordered_quad_is_valid(row.get("quad"))) for row in visible)
        matches = greedy_matches(visible, sample["targets"], similarity, 0.5)
        matched_targets = {target_index for _, target_index, _ in matches}
        tp += len(matches)
        for prediction_index, target_index, _ in matches:
            prediction = visible[prediction_index]["quad"]
            target = sample["targets"][target_index]
            distances = [math.dist(prediction[corner], target[corner]) for corner in range(4)]
            matched_errors.extend(distances)
            matched_normalized_errors.extend([value / max(math.sqrt(polygon_area(target)), 1.0) for value in distances])
        for index, target in enumerate(sample["targets"]):
            area_fraction = polygon_area(target) / max(float(sample["size"][0] * sample["size"][1]), 1.0)
            is_small = area_fraction < 0.01
            is_tiny = min(math.dist(target[corner], target[(corner + 1) % 4]) for corner in range(4)) < 16.0
            small_total += int(is_small)
            small_tp += int(is_small and index in matched_targets)
            tiny_total += int(is_tiny)
            tiny_tp += int(is_tiny and index in matched_targets)
    ap_levels = np.arange(0.5, 1.0, 0.05)
    return {
        "score_floor": score_floor,
        "ground_truth_targets": target_count,
        "predictions": predictions_count,
        "tp": tp,
        "fp": predictions_count - tp,
        "fn": target_count - tp,
        "precision50": tp / max(predictions_count, 1),
        "recall50": tp / max(target_count, 1),
        "polygon_AP50": average_precision(samples, similarity, 0.50, score_floor),
        "polygon_AP75": average_precision(samples, similarity, 0.75, score_floor),
        "polygon_AP50_95": float(np.mean([average_precision(samples, similarity, level, score_floor) for level in ap_levels])),
        "local_oks_AP50": average_precision(samples, local_oks, 0.50, score_floor),
        "local_oks_AP75": average_precision(samples, local_oks, 0.75, score_floor),
        "local_oks_AP50_95": float(np.mean([average_precision(samples, local_oks, level, score_floor) for level in ap_levels])),
        "small_targets": small_total,
        "small_tp": small_tp,
        "small_recall50": small_tp / max(small_total, 1),
        "tiny_targets": tiny_total,
        "tiny_tp": tiny_tp,
        "tiny_recall50": tiny_tp / max(tiny_total, 1),
        "corner_error_mean_px": float(np.mean(matched_errors)) if matched_errors else None,
        "corner_error_median_px": float(np.median(matched_errors)) if matched_errors else None,
        "corner_error_mean_normalized": float(np.mean(matched_normalized_errors)) if matched_normalized_errors else None,
        "invalid_geometry": invalid,
    }
