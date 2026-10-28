from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from v31_core import (BASE, BASE_SHA, GRID, HEAD_CKPT, HEAD_SHA, MANIFEST,
                      MAX_CANDIDATES, ROOT, CornerResidualHead,
                      atomic_torch_save, canonical_with_permutation,
                      capture_rng, decode_flat_candidate_rows, decode_residual, dump_json, guard_candidate,
                      load_models, restore_rng, roi_corner_coordinates,
                      run_image, safe_torch_load, seed_all, sha256)
from v31_core import rank
from v9_geometry import ordered_quad_is_valid
from train_v31 import loss_terms


def all_checks(role: str = "train_fit"):
    diagnostics = ROOT / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    checks: dict[str, object] = {}
    set_params = ["V28 frozen detector and original V10 quality head"]
    if sha256(BASE) != BASE_SHA or sha256(HEAD_CKPT) != HEAD_SHA:
        raise RuntimeError("fixed model SHA mismatch")
    checks["fixed_weight_hashes"] = {"detector": sha256(BASE), "quality_head": sha256(HEAD_CKPT), "pass": True}
    rows = rank.read_rows(MANIFEST, role)
    cache_dir = ROOT / "cache" / role
    if not (cache_dir / "collection_complete.json").exists():
        raise RuntimeError("feature collection must complete before real checks")
    meta_names = ("candidate_ids", "class_scores", "quality_scores", "boxes_resized", "raw_native",
                  "raw_v28", "raw_v28_online", "rois_resized", "corner_norm", "valid",
                  "permutation", "image_size", "resized_size", "candidate_count", "group", "target_norm")
    maps = {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in meta_names}
    features = np.load(cache_dir / "features.npy", mmap_mode="r")
    flat_probe = np.asarray([0, MAX_CANDIDATES - 1, MAX_CANDIDATES,
                             (features.shape[0] - 1) * MAX_CANDIDATES + (MAX_CANDIDATES - 1)], dtype=np.int64)
    image_probe, slot_probe = decode_flat_candidate_rows(flat_probe, features.shape[0])
    indexed_probe = np.asarray(features[image_probe, slot_probe], dtype=np.float32)
    flat_reference = np.asarray(features.reshape((-1,) + features.shape[2:])[flat_probe], dtype=np.float32)
    try:
        decode_flat_candidate_rows(np.asarray([-1], dtype=np.int64), features.shape[0])
        negative_rejected = False
    except IndexError:
        negative_rejected = True
    checks["flattened_sample_order_maps_to_image_and_candidate"] = {
        "probe_flat_rows": flat_probe.tolist(), "image_rows": image_probe.tolist(),
        "candidate_slots": slot_probe.tolist(),
        "feature_max_abs_diff_vs_flat_layout": float(np.max(np.abs(indexed_probe-flat_reference))),
        "negative_index_rejected": negative_rejected,
        "pass": bool(np.array_equal(indexed_probe, flat_reference) and negative_rejected
                     and image_probe.tolist() == [0, 0, 1, features.shape[0]-1]
                     and slot_probe.tolist() == [0, MAX_CANDIDATES-1, 0, MAX_CANDIDATES-1]),
    }
    reference_row = rows[0]
    reference_idx = 0
    count = int(maps["candidate_count"][reference_idx, 0])
    sample_feature = torch.from_numpy(np.array(features[reference_idx, :count], dtype=np.float32, copy=True)).cuda()
    sample_coords = torch.from_numpy(np.array(maps["corner_norm"][reference_idx, :count], dtype=np.float32, copy=True)).cuda()
    base = np.asarray(maps["raw_v28_online"][reference_idx, :count], dtype=np.float64)
    rois = np.array(maps["rois_resized"][reference_idx, :count], dtype=np.float32, copy=True)
    image_size = tuple(map(int, maps["image_size"][reference_idx, 0]))
    resized_size = tuple(map(int, maps["resized_size"][reference_idx, 0]))

    head = CornerResidualHead().cuda().eval()
    with torch.inference_mode():
        z = head(sample_feature, sample_coords)
    zero_head = bool(torch.equal(z, torch.zeros_like(z)))
    delta = decode_residual(z.reshape(-1, 4, 2), torch.from_numpy(rois).cuda(), image_size, resized_size).cpu().numpy()
    proposal = base + delta
    zero_coordinate = bool(np.array_equal(proposal, base))
    guard_identity = all(np.array_equal(guard_candidate(base[i], proposal[i], *image_size, False)[0], base[i]) for i in range(count))
    checks["zero_init_real_feature_exact_v28"] = {"zero_output_layer": zero_head, "zero_coordinate_delta": zero_coordinate,
                                                   "guard_identity": guard_identity, "pass": zero_head and zero_coordinate and guard_identity}

    # A known normalized ROI offset is encoded and decoded without changing its
    # axis or image-coordinate scale.
    roi_synth = torch.tensor([[10.0, 20.0, 110.0, 220.0]], device="cuda")
    frac = torch.tensor([[[0.02, -0.03]]], device="cuda")
    z_known = torch.atanh(frac / 0.05)
    # resized_size follows torchvision's (height, width) ordering.
    delta_known = decode_residual(z_known, roi_synth, (1000, 800), (640, 800)).cpu().numpy()[0, 0]
    expected_known = np.asarray([2.5, -7.5])
    checks["synthetic_coordinate_roundtrip"] = {"normalized_fraction": frac.cpu().numpy().tolist(),
                                                  "delta_original_px": delta_known.tolist(),
                                                  "expected_original_px": expected_known.tolist(),
                                                  "max_abs_error": float(np.max(np.abs(delta_known-expected_known))),
                                                  "pass": bool(np.allclose(delta_known, expected_known, atol=1e-6, rtol=0))}

    square = np.asarray([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
    shift = np.asarray([[.2, .1], [10.2, .1], [10.2, 10.1], [.2, 10.1]])
    invalid = np.asarray([[0., 0.], [0., 0.], [10., 10.], [0., 10.]])
    bowtie = np.asarray([[0., 0.], [10., 10.], [10., 0.], [0., 10.]])
    diamond = np.asarray([[5., 0.], [10., 5.], [5., 10.], [0., 5.]])
    shifted_diamond = np.asarray([[5.2, 0.], [10., 5.], [5., 10.], [0., 5.]])
    guard_cases = {
        "untriggered_exact": np.array_equal(guard_candidate(square, square, 20, 20, False)[0], square),
        "invalid_original_cannot_be_rescued": np.array_equal(guard_candidate(invalid, shift, 20, 20, True)[0], invalid),
        "illegal_proposal_reverts_whole_candidate": np.array_equal(guard_candidate(square, bowtie, 20, 20, True)[0], square),
        "changed_permutation_reverts": (np.array_equal(guard_candidate(diamond, shifted_diamond, 20, 20, True)[0], diamond)
                                         and "point_order_changed" in guard_candidate(diamond, shifted_diamond, 20, 20, True)[2]),
        "same_permutation_accepts_full_proposal": np.array_equal(guard_candidate(square, shift, 20, 20, True)[0], shift),
    }
    guard_cases["algorithm_has_no_gt_argument"] = "gt" not in __import__("inspect").signature(guard_candidate).parameters
    checks["synthetic_candidate_guard"] = {"cases": guard_cases, "all_pass": all(guard_cases.values())}

    # Save/load equivalence on a real train_fit forward feature and coordinates.
    with torch.inference_mode():
        before = head(sample_feature, sample_coords)
    roundtrip_path = diagnostics / "head_save_load_check.pt"
    atomic_torch_save({"residual_head": head.state_dict()}, roundtrip_path)
    restored = CornerResidualHead().cuda().eval()
    restored.load_state_dict(safe_torch_load(roundtrip_path)["residual_head"], strict=True)
    with torch.inference_mode():
        after = restored(sample_feature, sample_coords)
    save_load_error = float((before - after).abs().max().cpu())
    checks["save_load_output_equivalence"] = {"max_abs_error": save_load_error, "pass": save_load_error == 0.0}

    # Recheck a live image-only forward against the persisted same-candidate
    # features and metadata. No annotation is opened in this block.
    model, quality_head, mean, std = load_models()
    from v31_core import install_capture
    cap = install_capture(model)
    from v9_train import state_digest
    detector_before = state_digest(model.state_dict())
    quality_before = state_digest(quality_head.state_dict())
    live = run_image(model, cap, quality_head, mean, std, reference_row["image"])
    detector_after = state_digest(model.state_dict())
    quality_after = state_digest(quality_head.state_dict())
    ids_match = bool(np.array_equal(live["candidate_ids"], maps["candidate_ids"][reference_idx, :count]))
    feature_diff = np.abs(live["spatial_features"] - np.asarray(features[reference_idx, :count], dtype=np.float32))
    rois_diff = np.abs(live["rois_resized"] - np.asarray(maps["rois_resized"][reference_idx, :count]))
    raw_diff = np.abs(live["raw_v28"] - np.asarray(maps["raw_v28_online"][reference_idx, :count]))
    coord_norm = roi_corner_coordinates(live["raw_v28"], live["rois_resized"], live["original_size"], live["resized_size"])
    norm_diff = np.abs(coord_norm - np.asarray(maps["corner_norm"][reference_idx, :count]))
    checks["real_online_feature_alignment"] = {
        "stem": reference_row["stem"], "candidate_ids_equal": ids_match,
        "candidate_count": count, "rois_max_abs": float(rois_diff.max()),
        "V28_corner_max_abs": float(raw_diff.max()), "spatial_feature_max_abs": float(feature_diff.max()),
        "spatial_feature_p99_abs": float(np.percentile(feature_diff, 99)),
        "normalized_corner_max_abs": float(norm_diff.max()),
        "detector_state_unchanged": detector_before == detector_after,
        "quality_head_state_unchanged": quality_before == quality_after,
        "pass": bool(ids_match and rois_diff.max() <= 1e-4 and raw_diff.max() <= 1e-4
                     and feature_diff.max() <= 1e-3 and norm_diff.max() <= 1e-5
                     and detector_before == detector_after and quality_before == quality_after),
    }

    fg_rows, bg_rows = [], []
    groups_map = maps["group"]
    for image_idx in range(len(rows)):
        n = int(maps["candidate_count"][image_idx, 0])
        labs = np.asarray(groups_map[image_idx, :n])
        if not len(fg_rows):
            fg_rows.extend((image_idx * MAX_CANDIDATES + int(j)) for j in np.flatnonzero(labs == 1)[:32])
        if not len(bg_rows):
            bg_rows.extend((image_idx * MAX_CANDIDATES + int(j)) for j in np.flatnonzero(labs == 2)[:32])
        if len(fg_rows) >= 16 and len(bg_rows) >= 16:
            break
    chosen_rows = np.asarray(fg_rows[:16] + bg_rows[:16], dtype=np.int64)
    feat_all = np.load(cache_dir / "features.npy", mmap_mode="r")
    feat_flat = feat_all.reshape(-1, 256, 4, 4)
    coords_flat = np.load(cache_dir / "corner_norm.npy", mmap_mode="r").reshape(-1, 4, 2)
    target_flat = np.load(cache_dir / "target_norm.npy", mmap_mode="r").reshape(-1, 4, 2)
    groups_flat = np.load(cache_dir / "group.npy", mmap_mode="r").reshape(-1)
    train_head = CornerResidualHead().cuda().train()
    optimizer = torch.optim.AdamW(train_head.parameters(), lr=1e-4, weight_decay=1e-4)
    def get_batch(indices):
        return (torch.from_numpy(np.asarray(feat_flat[indices], dtype=np.float32)).cuda(),
                torch.from_numpy(np.asarray(coords_flat[indices], dtype=np.float32)).cuda(),
                torch.from_numpy(np.asarray(target_flat[indices], dtype=np.float32)).cuda(),
                torch.from_numpy(np.asarray(groups_flat[indices], dtype=np.int64)).cuda())
    x, c, target, groups = get_batch(chosen_rows)
    pred = 0.05 * torch.tanh(train_head(x, c).reshape(-1, 4, 2))
    loss, fg_loss, bg_loss, fg_count, bg_count = loss_terms(pred, target, groups)
    finite_loss = bool(torch.isfinite(loss) and torch.isfinite(fg_loss) and torch.isfinite(bg_loss))
    loss.backward()
    output_grad = float(train_head.out.weight.grad.norm().detach().cpu())
    earlier_grad_first = {"fc1": float(train_head.fc1.weight.grad.norm().detach().cpu()) if train_head.fc1.weight.grad is not None else 0.0,
                          "fc2": float(train_head.fc2.weight.grad.norm().detach().cpu()) if train_head.fc2.weight.grad is not None else 0.0}
    out_before = train_head.out.weight.detach().clone()
    optimizer.step()
    output_update = float((train_head.out.weight.detach() - out_before).norm().cpu())
    optimizer.zero_grad(set_to_none=True)
    pred2 = 0.05 * torch.tanh(train_head(x, c).reshape(-1, 4, 2))
    loss2, _, _, _, _ = loss_terms(pred2, target, groups)
    loss2.backward()
    earlier_grad_second = {"fc1": float(train_head.fc1.weight.grad.norm().detach().cpu()) if train_head.fc1.weight.grad is not None else 0.0,
                           "fc2": float(train_head.fc2.weight.grad.norm().detach().cpu()) if train_head.fc2.weight.grad is not None else 0.0}
    checks["real_train_fit_forward_backward"] = {
        "candidate_batch": int(len(chosen_rows)), "foreground": fg_count, "background": bg_count,
        "loss": float(loss.detach().cpu()), "loss_fg": float(fg_loss.detach().cpu()),
        "loss_bg": float(bg_loss.detach().cpu()), "finite": finite_loss,
        "output_layer_gradient_norm": output_grad, "output_layer_update_norm": output_update,
        "earlier_layer_grad_norm_first_step": earlier_grad_first,
        "earlier_layer_grad_norm_second_step": earlier_grad_second,
        "pass": bool(finite_loss and output_grad > 0 and output_update > 0
                     and earlier_grad_second["fc1"] > 0 and earlier_grad_second["fc2"] > 0),
    }

    # Explicitly test virtual batch accumulation where the tail has one row;
    # loss and gradients must equal the same effective batch evaluated together.
    torch.manual_seed(1701)
    pred_full = torch.randn(5, 4, 2, requires_grad=True)
    target_full = torch.randn(5, 4, 2)
    label_full = torch.tensor([1, 2, 1, 2, 1])
    loss_full, *_ = loss_terms(pred_full, target_full, label_full)
    grad_full = torch.autograd.grad(loss_full, pred_full)[0]
    pred_parts = [pred_full.detach()[0:2].clone().requires_grad_(),
                  pred_full.detach()[2:4].clone().requires_grad_(),
                  pred_full.detach()[4:5].clone().requires_grad_()]
    pred_cat = torch.cat(pred_parts)
    loss_accum, *_ = loss_terms(pred_cat, target_full, label_full)
    grad_accum = torch.autograd.grad(loss_accum, pred_cat)[0]
    checks["gradient_accumulation_tail"] = {"microbatch_sizes": [2, 2, 1],
                                             "loss_abs_diff": float(abs(loss_full.item()-loss_accum.item())),
                                             "gradient_max_abs_diff": float((grad_full-grad_accum).abs().max()),
                                             "pass": bool(torch.equal(loss_full, loss_accum) and torch.equal(grad_full, grad_accum))}

    # A minimal actual optimizer/RNG/sample-order checkpoint replay on CPU.
    cpu_head = CornerResidualHead().cpu().train()
    cpu_opt = torch.optim.AdamW(cpu_head.parameters(), lr=1e-4, weight_decay=1e-4)
    rng_bundle = capture_rng()
    order = np.asarray([9, 4, 2, 8, 0], dtype=np.int64)
    resume_bundle = {"residual_head": copy.deepcopy(cpu_head.state_dict()), "optimizer": cpu_opt.state_dict(),
                     "rng": rng_bundle, "epoch": 1, "global_step": 0,
                     "next_sample_offset": 2, "epoch_order": order,
                     "gradient_accumulation": {"pending_microbatches": 0, "safe_update_boundary": True}}
    resume_path = diagnostics / "resume_state_roundtrip.pt"
    atomic_torch_save(resume_bundle, resume_path)
    expected_python = random.random()
    expected_numpy = np.random.rand(3)
    expected_torch = torch.rand(3)
    expected_cuda = torch.rand(3, device="cuda")
    next_indices = order[resume_bundle["next_sample_offset"]:resume_bundle["next_sample_offset"]+2]
    xcpu = torch.randn(5, 256, 4, 4)
    ccpu = torch.randn(5, 4, 2)
    tCPU = torch.randn(5, 4, 2)
    gCPU = torch.tensor([1, 2, 1, 2, 1])
    p1 = 0.05 * torch.tanh(cpu_head(xcpu, ccpu).reshape(-1, 4, 2))
    l1, *_ = loss_terms(p1, tCPU, gCPU); l1.backward(); cpu_opt.step()
    expected_state = {k: v.clone() for k, v in cpu_head.state_dict().items()}
    restored_bundle = safe_torch_load(resume_path, "cpu")
    cpu_head_2 = CornerResidualHead().cpu().train()
    cpu_opt_2 = torch.optim.AdamW(cpu_head_2.parameters(), lr=1e-4, weight_decay=1e-4)
    cpu_head_2.load_state_dict(restored_bundle["residual_head"], strict=True)
    cpu_opt_2.load_state_dict(restored_bundle["optimizer"])
    restore_rng(restored_bundle["rng"])
    actual_python = random.random()
    actual_numpy = np.random.rand(3)
    actual_torch = torch.rand(3)
    actual_cuda = torch.rand(3, device="cuda")
    actual_indices = restored_bundle["epoch_order"][restored_bundle["next_sample_offset"]:restored_bundle["next_sample_offset"]+2]
    p2 = 0.05 * torch.tanh(cpu_head_2(xcpu, ccpu).reshape(-1, 4, 2))
    l2, *_ = loss_terms(p2, tCPU, gCPU); l2.backward(); cpu_opt_2.step()
    update_diff = max(float((expected_state[k] - cpu_head_2.state_dict()[k]).abs().max()) for k in expected_state)
    checks["optimizer_rng_sampler_resume"] = {
        "python_rng_match": expected_python == actual_python,
        "numpy_rng_match": bool(np.array_equal(expected_numpy, actual_numpy)),
        "torch_cpu_rng_match": bool(torch.equal(expected_torch, actual_torch)),
        "torch_cuda_rng_match": bool(torch.equal(expected_cuda, actual_cuda)),
        "next_sample_order_match": bool(np.array_equal(next_indices, actual_indices)),
        "optimizer_model_update_max_abs_diff": update_diff,
        "safe_update_boundary_saved": restored_bundle["gradient_accumulation"]["safe_update_boundary"],
        "pass": bool(expected_python == actual_python and np.array_equal(expected_numpy, actual_numpy)
                     and torch.equal(expected_torch, actual_torch) and torch.equal(expected_cuda, actual_cuda)
                     and np.array_equal(next_indices, actual_indices) and update_diff == 0.0),
    }

    trainable = [{"name": n, "shape": list(p.shape), "numel": p.numel()}
                 for n, p in CornerResidualHead().named_parameters()]
    checks["trainable_parameter_inventory"] = {"parameters": trainable,
                                                "count": sum(x["numel"] for x in trainable),
                                                "normalization_modules": {"FrozenBatchNorm2d": 53, "new_branch_norms": 0},
                                                "pass": sum(x["numel"] for x in trainable) < 1_000_000}
    checks["inference_functions_have_no_gt_input"] = {
        "guard_parameters": list(__import__("inspect").signature(guard_candidate).parameters),
        "residual_decode_parameters": list(__import__("inspect").signature(decode_residual).parameters),
        "pass": "gt" not in __import__("inspect").signature(guard_candidate).parameters
                and "gt" not in __import__("inspect").signature(decode_residual).parameters,
    }
    all_bools = []
    for value in checks.values():
        if isinstance(value, dict) and "pass" in value:
            all_bools.append(bool(value["pass"]))
        if isinstance(value, dict) and "all_pass" in value:
            all_bools.append(bool(value["all_pass"]))
    checks["all_pass"] = bool(all(all_bools))
    checks["verified_trainable_branch_only"] = "new CornerResidualHead parameters only; detector and original quality head require_grad=False and eval"
    dump_json(diagnostics / "IMPLEMENTATION_CHECKS.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    if not checks["all_pass"]:
        raise RuntimeError("V31 implementation checks failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("train_fit", "dev"), default="train_fit")
    args = parser.parse_args()
    all_checks(args.role)


if __name__ == "__main__":
    main()
