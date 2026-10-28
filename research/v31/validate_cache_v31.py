from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v31_core import MANIFEST, ROOT, sha256


EXPECTED = {"train_fit": 7731, "dev": 859}
FINITE_MAPS = ("features", "class_scores", "quality_scores", "boxes_original", "boxes_resized",
               "raw_native", "raw_v28", "raw_v28_online", "rois_resized", "corner_norm",
               "fit_offset_resized", "target_norm")


def main():
    result = {"status": "checking", "manifest_sha256": sha256(MANIFEST),
              "roles": {}, "test300_read": False, "validation_locked_confirm_read": False,
              "dev_annotations_read": False}
    for role, expected_images in EXPECTED.items():
        directory = ROOT / "cache" / role
        complete_path = directory / "collection_complete.json"
        if not complete_path.exists():
            raise FileNotFoundError(complete_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "complete" or complete.get("images") != expected_images:
            raise RuntimeError(f"incomplete cache for {role}: {complete}")
        if complete.get("manifest_sha256") != result["manifest_sha256"]:
            raise RuntimeError(f"manifest identity mismatch for {role}")
        counts = np.load(directory / "candidate_count.npy", mmap_mode="r")[:, 0]
        if len(counts) != expected_images or not np.all(counts == 100):
            raise RuntimeError(f"unexpected candidate count distribution for {role}")
        ids = np.load(directory / "candidate_ids.npy", mmap_mode="r")
        for i in range(expected_images):
            n = int(counts[i])
            if len(np.unique(ids[i, :n])) != n:
                raise RuntimeError(f"duplicate candidate ID in {role} image index {i}")
        finite_summary = {}
        for name in FINITE_MAPS:
            path = directory / f"{name}.npy"
            array = np.load(path, mmap_mode="r")
            nonfinite = 0
            for start in range(0, expected_images, 16):
                block = np.asarray(array[start:start + 16])
                nonfinite += int((~np.isfinite(block)).sum())
            finite_summary[name] = {"nonfinite_values": nonfinite, "shape": list(array.shape),
                                    "dtype": str(array.dtype)}
            if nonfinite:
                raise FloatingPointError(f"nonfinite values in {role}/{name}: {nonfinite}")
        role_record = {"images": expected_images, "candidate_count_distribution": {"100": expected_images},
                       "candidate_ids_unique_per_image": True, "maps": finite_summary,
                       "frozen_detector_state_unchanged":
                           complete.get("frozen_detector_state_sha256_before") == complete.get("frozen_detector_state_sha256_after"),
                       "collection_summary": complete}
        if not role_record["frozen_detector_state_unchanged"]:
            raise RuntimeError(f"frozen detector changed during {role} collection")
        if role == "train_fit":
            supervision = json.loads((directory / "supervision_summary.json").read_text(encoding="utf-8"))
            groups = np.load(directory / "group.npy", mmap_mode="r")
            actual = {"foreground": int(np.sum(groups == 1)), "background": int(np.sum(groups == 2)),
                      "ignored": int(np.sum(groups < 0))}
            if actual != supervision["group_counts"]:
                raise RuntimeError(f"train_fit supervision totals differ: {actual} != {supervision['group_counts']}")
            role_record["supervision"] = supervision
            role_record["supervision_totals_match_memmap"] = True
        else:
            equivalence = json.loads((directory / "v28_online_equivalence.json").read_text(encoding="utf-8"))
            if equivalence.get("images") != expected_images:
                raise RuntimeError("dev V28 online equivalence record is incomplete")
            if not all(row.get("same_ids") and row.get("fit_state_equal") for row in equivalence["checks"]):
                raise RuntimeError("dev candidate or V28 fit_state differs from frozen reference")
            role_record["v28_online_equivalence"] = {
                "images": equivalence["images"], "all_candidate_ids_equal": True,
                "all_fit_states_equal": True, "max_box_abs": equivalence["max_box_abs"],
                "max_native_abs": equivalence["max_native_abs"], "max_v28_abs_diagnostic_only": equivalence["max_v28_abs"],
                "max_quality_vs_previous_online": equivalence["max_quality_vs_previous_online"],
                "max_quality_vs_frozen_cache": equivalence["max_quality_vs_frozen_cache"]}
        result["roles"][role] = role_record
    result["status"] = "pass"
    output = ROOT / "diagnostics" / "CACHE_INTEGRITY.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "cache_integrity", "status": result["status"],
                      "train_fit_images": EXPECTED["train_fit"], "dev_images": EXPECTED["dev"],
                      "dev_max_v28_online_abs_diagnostic": result["roles"]["dev"]["v28_online_equivalence"]["max_v28_abs_diagnostic_only"],
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
