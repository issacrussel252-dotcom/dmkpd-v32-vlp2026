from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(r"<PROJECT_ROOT>")
MANIFEST = PROJECT / "07_第5版_R6正式复跑_20261006" / "inputs" / "split_manifest_r2_internal.csv"
V28 = PROJECT / "DMKPD-V28-V10亚像素解码-配对复核"
V30 = PROJECT / "DMKPD-V30-V28受限单轴修正"
DETECTOR = PROJECT / "DMKPD-V9-20261009" / "train" / "A_main" / "epoch_12.pt"
QUALITY = PROJECT / "DMKPD-V10-20261010" / "train" / "geometry_seed20261010" / "epoch_10.pt"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    config_path = ROOT / "RUN_CONFIG.json"
    provenance_path = ROOT / "PROVENANCE.json"
    if config_path.exists() or provenance_path.exists():
        raise FileExistsError("frozen config/provenance already exists; do not overwrite it")
    required = [
        DETECTOR, QUALITY, MANIFEST,
        V28 / "FINAL_REPORT.md", V28 / "PREREGISTRATION.md", V28 / "RUN_CONFIG.json",
        V28 / "RESULTS.json", V28 / "RUNTIME.json", V28 / "results" / "B_predictions.json",
        V28 / "code" / "run_v28_fixed.py",
        V28 / "code" / "source_snapshot" / "v10_rank.py",
        V28 / "code" / "source_snapshot" / "v9_geometry.py",
        V28 / "code" / "source_snapshot" / "v9_evaluate.py",
        V28 / "code" / "source_snapshot" / "metric_helpers.py",
        V30 / "FINAL_REPORT.md", V30 / "PLAN.md", V30 / "RUN_CONFIG.json", V30 / "code" / "run_v30.py",
        ROOT / "diagnostics" / "RESOURCE_PREFLIGHT.json", ROOT / "logs" / "IMPLEMENTATION_REPAIR.json",
        ROOT / "PLAN.md", ROOT / "METHOD_REFERENCES.md",
        ROOT / "NUMERICAL_COLLECTION_ADDENDUM.md",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing required provenance inputs: " + "; ".join(missing))
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    selected = [r for r in manifest_rows if r.get("split") in ("train_fit", "dev")]
    counts = {role: sum(r.get("split") == role for r in selected) for role in ("train_fit", "dev")}
    if counts != {"train_fit": 7731, "dev": 859}:
        raise RuntimeError(f"split counts changed: {counts}")

    code_files = sorted(p for p in (ROOT / "code").rglob("*.py") if "__pycache__" not in p.parts)
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_files}
    key_hashes = {str(p): sha(p) for p in required}

    # Hash only train_fit labels used for supervision. Dev annotations are read
    # by the evaluator after its predictions are constructed and are not opened
    # by this pretraining freeze operation.
    dataset_hashes = []
    for i, row in enumerate(selected, 1):
        role = row["split"]
        image = Path(row["image"])
        if not image.is_file():
            raise FileNotFoundError(image)
        actual_image_sha = sha(image)
        declared_sha = row.get("sha256", "")
        if declared_sha and actual_image_sha.lower() != declared_sha.lower():
            raise RuntimeError(f"manifest image hash mismatch for {row['stem']}: {actual_image_sha} != {declared_sha}")
        entry = {"role": role, "stem": row["stem"], "image": str(image),
                 "image_sha256": actual_image_sha, "manifest_sha256_field": declared_sha}
        if role == "train_fit":
            annotation = Path(row["annotation"])
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            entry["train_fit_annotation"] = str(annotation)
            entry["train_fit_annotation_sha256"] = sha(annotation)
        dataset_hashes.append(entry)
        if i % 1000 == 0:
            print(json.dumps({"stage": "hash_inputs", "rows": i, "total": len(selected)}, ensure_ascii=False), flush=True)

    cache_files = sorted((V28 / "cache" / "dev").glob("*.npz"))
    if len(cache_files) != 859:
        raise RuntimeError(f"expected 859 frozen V28 dev cache files, found {len(cache_files)}")
    cache_manifest = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in cache_files]
    cache_digest = hashlib.sha256(json.dumps(cache_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    hashes_path = ROOT / "DATA_INPUT_HASHES.json"
    write(hashes_path, {"manifest_sha256": sha(MANIFEST), "manifest_split_counts": counts,
                        "train_fit_and_dev_images_and_train_fit_annotations": dataset_hashes,
                        "V28_dev_cache": cache_manifest, "V28_dev_cache_inventory_sha256": cache_digest,
                        "dev_annotations_hashed_before_predictions": False})

    config = {
        "experiment": "DMKPD V31 learned continuous corner residual head",
        "frozen_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hypothesis": "freeze V28 detector/keypoint and V10 quality route; train one candidate-specific spatial RoI residual head",
        "arms": {"A": "exact frozen V28 output", "B": "V28 plus one learned bounded corner residual head with V30 candidate-level validity/permutation guard"},
        "fixed_main_epoch": 3,
        "seed": 20261031,
        "model": {"detector_path": str(DETECTOR), "detector_sha256": sha(DETECTOR),
                  "quality_head_path": str(QUALITY), "quality_head_sha256": sha(QUALITY),
                  "normalization": "FrozenBatchNorm2d", "expected_frozen_batchnorm_count": 53},
        "data": {"manifest_path": str(MANIFEST), "manifest_sha256": sha(MANIFEST),
                 "train_fit_images": 7731, "train_fit_gt": "not counted in the frozen config",
                 "dev_images": 859, "dev_gt": 1981,
                 "supervision_annotations": "train_fit only", "dev_role": "evaluation only",
                 "validation_locked_confirm_test300_read": False},
        "new_head": {"input_roi_shape": [256, 14, 14], "adaptive_spatial_pool": [4, 4],
                     "corner_coordinate_features": 8, "layers": [4104, 128, 64, 8],
                     "activation": "ReLU", "output_init": "zero weight and bias",
                     "parameters": 534216, "max_roi_fraction_per_axis": 0.05,
                     "output_coordinate_frame": "raw keypoint channel order, original pixels"},
        "training": {"epochs": 3, "optimizer": "AdamW", "learning_rate": 1e-4,
                     "weight_decay": 1e-4, "microbatch": 256, "gradient_accumulation": 2,
                     "effective_batch_max": 512, "background_per_image_cap": "min(background_count, foreground_count)",
                     "augmentation": "none", "loss": "mean SmoothL1 foreground + 0.1 * mean SmoothL1 background-to-zero"},
        "inference": {"input_short_side": 768, "max_long_side": 1024, "dtype": "FP32", "batch_size": 1,
                      "pre_keypoint_candidate_budget": 100, "polygon_nms_iou": 0.30,
                      "final_output_cap": 100, "primary_AP_score_floor": None,
                      "working_point_floors": [0.30, 0.45],
                      "quality_features_from_unmodified_V10_corners": True,
                      "quality_score_formula_unchanged": True},
        "evaluation": {"A_reference_local_oks_ap50_95": 0.7905144222533753,
                       "A_reference_polygon_ap50_95": 0.8181647489075556,
                       "A_reference_floor030": {"tp": 1767, "fp": 198, "fn": 214,
                                                "small_detected": 612, "small_targets": 699,
                                                "tiny_detected": 20, "tiny_targets": 31,
                                                "boundary_detected": 23, "boundary_targets": 26},
                       "bootstrap": {"unit": "image", "replicates": 1000, "seed": 20261031},
                       "promotion_rule": "frozen in PLAN.md; do full online dev only if every gate passes"},
        "online_collection_equivalence": {"candidate_id_exact": True,
                                          "box_and_native_corner_tolerance_px": 1e-4,
                                          "fit_state_exact": True,
                                          "refined_corner_difference": "recorded as numerical diagnostic; not used to select candidates or alter A/B cache evaluation",
                                          "numerical_addendum": str(ROOT / "NUMERICAL_COLLECTION_ADDENDUM.md")},
        "resource": {"gpu_active_hours_hard_cap": 8, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        "source_hashes": key_hashes,
        "v31_code_hashes": code_hashes,
        "data_input_hashes_path": str(hashes_path),
        "data_input_hashes_sha256": sha(hashes_path),
        "external_method_access_date": "2026-10-25",
        "test300_read": False, "submission_created": False, "upload": False,
    }
    write(config_path, config)
    write(provenance_path, {"run_config_sha256": sha(config_path), "plan_sha256": sha(ROOT / "PLAN.md"),
                            "method_references_sha256": sha(ROOT / "METHOD_REFERENCES.md"),
                            "data_input_hashes_sha256": sha(hashes_path),
                            "input_sha256": key_hashes, "code_sha256": code_hashes,
                            "python": sys.version, "platform": platform.platform(),
                            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                            "frozen_timestamp_local": config["frozen_at_local"],
                            "dev_annotations_hashed_before_predictions": False,
                            "test300_read": False})
    print(json.dumps({"stage": "frozen_config_written", "run_config_sha256": sha(config_path),
                      "data_input_hashes_sha256": sha(hashes_path), "hashed_data_rows": len(dataset_hashes),
                      "v28_cache_files": len(cache_files), "code_files": len(code_files)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
