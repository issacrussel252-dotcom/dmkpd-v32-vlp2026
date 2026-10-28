from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"<PROJECT_ROOT>\DMKPD-V31-V28学习式角点残差")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "source_snapshot"))
sys.path.insert(0, str(ROOT / "code" / "source_refs"))
os.environ["DMKPD_V32_RUN_ROOT"] = str(ROOT)
os.environ["DMKPD_V32_SOURCE_ROOT"] = str(SOURCE)
from v31_core import CornerResidualHead, capture_rng, seed_all, set_v28_backend, sha256

SEEDS = (20261031, 20261101)
EPOCHS = (1, 2, 3)
MAX_CANDIDATES = 100


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_order(group, count, seed, epoch):
    rng = np.random.default_rng(seed + 10007 * epoch)
    image_order = rng.permutation(len(count))
    selected = []
    coverage = {"images_seen": 0, "images_with_foreground": 0, "images_with_background": 0,
                "foreground_selected": 0, "background_pool": 0, "background_selected": 0,
                "ignored_not_selected": 0}
    for image_idx in image_order:
        n = int(count[image_idx, 0])
        groups = np.asarray(group[image_idx, :n])
        fg = np.flatnonzero(groups == 1)
        bg = np.flatnonzero(groups == 2)
        coverage["images_seen"] += 1
        coverage["images_with_foreground"] += int(len(fg) > 0)
        coverage["images_with_background"] += int(len(bg) > 0)
        coverage["foreground_selected"] += len(fg)
        coverage["background_pool"] += len(bg)
        bg_n = min(len(bg), len(fg))
        bg = rng.choice(bg, size=bg_n, replace=False) if bg_n else np.empty(0, dtype=np.int64)
        local = np.concatenate((fg.astype(np.int64), bg.astype(np.int64)))
        rng.shuffle(local)
        selected.extend(int(image_idx) * MAX_CANDIDATES + int(j) for j in local)
        coverage["background_selected"] += len(bg)
        coverage["ignored_not_selected"] += int(np.sum(groups < 0)) + max(0, int(np.sum(groups == 2)) - len(bg))
    if coverage["background_selected"] > coverage["foreground_selected"]:
        raise RuntimeError("background cap exceeded")
    return np.asarray(selected, dtype=np.int64), coverage, rng.bit_generator.state


def main():
    cache = SOURCE / "cache" / "train_fit"
    integrity = json.loads((SOURCE / "diagnostics" / "CACHE_INTEGRITY.json").read_text(encoding="utf-8"))
    complete = json.loads((cache / "collection_complete.json").read_text(encoding="utf-8"))
    if integrity.get("status") != "pass" or complete.get("status") != "complete":
        raise RuntimeError("V31 train_fit cache integrity/collection status not passing")
    features = np.load(cache / "features.npy", mmap_mode="r")
    group = np.load(cache / "group.npy", mmap_mode="r")
    target = np.load(cache / "target_norm.npy", mmap_mode="r")
    count = np.load(cache / "candidate_count.npy", mmap_mode="r")
    small = np.load(cache / "matched_small.npy", mmap_mode="r")
    tiny = np.load(cache / "matched_tiny.npy", mmap_mode="r")
    if features.shape != (7731, 100, 256, 4, 4) or group.shape != (7731, 100) or target.shape != (7731, 100, 4, 2):
        raise RuntimeError(f"cache shape mismatch: features={features.shape}, group={group.shape}, target={target.shape}")
    if not np.isfinite(np.asarray(target)).all():
        raise RuntimeError("nonfinite residual targets")

    fg_rows = np.flatnonzero(np.asarray(group).reshape(-1) == 1)
    fg_values = np.asarray(target.reshape(-1, 8)[fg_rows], dtype=np.float64).reshape(-1)
    ae = np.abs(fg_values)
    g_a = np.minimum(ae / 1.0, 1.0)
    g_b = np.minimum(ae / 0.01, 1.0)
    rms_a = math.sqrt(float(np.mean(g_a * g_a, dtype=np.float64)))
    rms_b = math.sqrt(float(np.mean(g_b * g_b, dtype=np.float64)))
    alpha = rms_a / rms_b
    top_n = int(math.ceil(len(ae) * 0.01))
    top_ix = np.argpartition(np.abs(g_a), len(ae) - top_n)[-top_n:]
    stats = {
        "foreground_candidate_rows": int(len(fg_rows)),
        "foreground_coordinate_components": int(len(fg_values)),
        "abs_target_median": float(np.median(ae)),
        "abs_target_fraction_gt_0_05": float(np.mean(ae > 0.05)),
        "beta1_output_gradient_rms": rms_a,
        "beta0_01_output_gradient_rms_unscaled": rms_b,
        "alpha": alpha,
        "after_scale_beta0_01_rms": alpha * rms_b,
        "rms_abs_error": float(np.sqrt(np.mean(fg_values * fg_values, dtype=np.float64))),
        "top_1pct_abs_error_gradient_share": float(np.sum(np.abs(g_a[top_ix]), dtype=np.float64) / np.sum(np.abs(g_a), dtype=np.float64)),
        "top_1pct_component_count": top_n,
        "qualification": "candidate coordinate components, not independent GTs or parameter gradients",
        "formula": "RMS(sign(e)*min(abs(e)/1,1))/RMS(sign(e)*min(abs(e)/0.01,1)) at pred=0",
    }
    dump(ROOT / "ALPHA_CALIBRATION.json", stats)

    input_hashes = {}
    for name in ("features.npy", "group.npy", "target_norm.npy", "corner_norm.npy", "rois_resized.npy",
                 "candidate_count.npy", "matched_small.npy", "matched_tiny.npy", "collection_complete.json"):
        p = cache / name
        input_hashes[str(p)] = sha256(p)
    dump(ROOT / "CACHE_VALIDATION.json", {"integrity_record": str(SOURCE / "diagnostics" / "CACHE_INTEGRITY.json"),
         "integrity_record_sha256": sha256(SOURCE / "diagnostics" / "CACHE_INTEGRITY.json"),
         "collection_status": complete, "train_fit_images": int(features.shape[0]),
         "candidate_rows": int(features.shape[0] * features.shape[1]), "target_statistics": {
             "foreground_rows": int(len(fg_rows)), "background_rows": int(np.sum(group == 2)),
             "ignored_rows": int(np.sum(group < 0))}, "input_hashes": input_hashes,
         "no_feature_reextraction": True})

    order_manifest = {}
    for seed in SEEDS:
        order_manifest[str(seed)] = {}
        for epoch in EPOCHS:
            arr, coverage, rng_state = make_order(group, count, seed, epoch)
            dst = ROOT / "sampling_orders" / f"seed_{seed}" / f"epoch_{epoch:02d}.npy"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, arr)
            source_match = None
            if seed == 20261031:
                old = np.load(cache / f"epoch_{epoch:02d}_sample_order.npy", mmap_mode="r")
                source_match = bool(np.array_equal(arr, old))
                if not source_match:
                    raise RuntimeError(f"reconstructed source seed sample order mismatch at epoch {epoch}")
            record = {"epoch": epoch, "seed": seed + 10007 * epoch, "num_rows": int(len(arr)),
                      "coverage": coverage, "sample_order_path": str(dst), "sample_order_sha256": sha256(dst),
                      "source_v31_order_exact_match": source_match, "sampler_rng_state": rng_state}
            dump(dst.with_suffix(".json"), record)
            order_manifest[str(seed)][str(epoch)] = record
    dump(ROOT / "SAMPLING_MANIFEST.json", order_manifest)

    init_manifest = {}
    for seed in SEEDS:
        set_v28_backend()
        seed_all(seed)
        head = CornerResidualHead()
        state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if not torch.equal(state["out.weight"], torch.zeros_like(state["out.weight"])) or not torch.equal(state["out.bias"], torch.zeros_like(state["out.bias"])):
            raise RuntimeError("residual output layer was not zero initialized")
        path = ROOT / "initialization" / f"seed_{seed}_initial.pt"
        torch.save({"seed": seed, "state_dict": state, "post_init_rng": capture_rng(),
                    "strictly_new_initialization": True}, path)
        init_manifest[str(seed)] = {"path": str(path), "sha256": sha256(path),
                                    "state_tensor_shapes": {k: list(v.shape) for k, v in state.items()},
                                    "parameter_count": sum(v.numel() for v in state.values()),
                                    "output_zero_initialized": True}
    dump(ROOT / "INITIALIZATION_HASHES.json", init_manifest)

    # Fixed, balanced diagnostic candidates, sampled only from train_fit groups.
    d_rng = np.random.default_rng(20261102)
    flat_group = np.asarray(group).reshape(-1)
    fg_pool = np.flatnonzero(flat_group == 1)
    bg_pool = np.flatnonzero(flat_group == 2)
    chosen_fg = d_rng.choice(fg_pool, size=256, replace=False)
    chosen_bg = d_rng.choice(bg_pool, size=256, replace=False)
    diag = np.concatenate((chosen_fg, chosen_bg)).astype(np.int64)
    d_rng.shuffle(diag)
    diag_path = ROOT / "diagnostics" / "fixed_train_fit_diagnostic_rows.npy"
    np.save(diag_path, diag)
    dump(ROOT / "diagnostics" / "FIXED_DIAGNOSTIC_BATCH.json", {
        "seed": 20261102, "candidate_count": int(len(diag)), "foreground": 256, "background": 256,
        "row_indices_sha256": sha256(diag_path), "source": "train_fit cache only",
        "small_count": int(np.sum(small.reshape(-1)[chosen_fg] > 0)),
        "tiny_count": int(np.sum(tiny.reshape(-1)[chosen_fg] > 0)),
        "grouping_fixed_before_dev": True})
    print(json.dumps({"status": "prepared", "alpha": alpha, "cache_hash_count": len(input_hashes),
                      "sample_order_checks": [order_manifest["20261031"][str(e)]["source_v31_order_exact_match"] for e in EPOCHS],
                      "inits": init_manifest}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
