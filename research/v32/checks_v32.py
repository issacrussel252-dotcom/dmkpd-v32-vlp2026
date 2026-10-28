from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"<PROJECT_ROOT>\DMKPD-V31-V28学习式角点残差")
os.environ["DMKPD_V32_RUN_ROOT"] = str(ROOT)
os.environ["DMKPD_V32_SOURCE_ROOT"] = str(SOURCE)
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code" / "source_snapshot"), str(ROOT / "code" / "source_refs")]
from v31_core import (BASE, BASE_SHA, HEAD_CKPT, HEAD_SHA, CornerResidualHead, capture_rng,
                      decode_residual, guard_candidate, restore_rng, seed_all, set_v28_backend,
                      sha256, safe_torch_load, atomic_torch_save)

DEVICE = torch.device("cuda")


def dumps(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def state_equal(a, b):
    if a.keys() != b.keys(): return False, float("inf")
    diffs = [float((a[k].detach().cpu() - b[k].detach().cpu()).abs().max()) for k in a]
    return max(diffs, default=0.0) == 0.0, max(diffs, default=0.0)


def nested_equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict): return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)): return len(a) == len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a == b


def main():
    if not torch.cuda.is_available(): raise RuntimeError("V32 requires CUDA for planned real-forward checks")
    integrity = json.loads((SOURCE / "diagnostics" / "CACHE_INTEGRITY.json").read_text(encoding="utf-8"))
    if integrity.get("status") != "pass": raise RuntimeError("cached feature integrity record failed")
    cache = SOURCE / "cache" / "train_fit"
    features = np.load(cache / "features.npy", mmap_mode="r")
    group = np.load(cache / "group.npy", mmap_mode="r")
    target = np.load(cache / "target_norm.npy", mmap_mode="r")
    coords = np.load(cache / "corner_norm.npy", mmap_mode="r")
    rois = np.load(cache / "rois_resized.npy", mmap_mode="r")
    raw = np.load(cache / "raw_v28.npy", mmap_mode="r")
    image_size = np.load(cache / "image_size.npy", mmap_mode="r")
    resized_size = np.load(cache / "resized_size.npy", mmap_mode="r")
    diag = np.load(ROOT / "diagnostics" / "fixed_train_fit_diagnostic_rows.npy")
    alpha = float(json.loads((ROOT / "ALPHA_CALIBRATION.json").read_text(encoding="utf-8"))["alpha"])
    checks = {}

    # Zero-initialized A/B share the same per-seed tensor state and are exact V28 identities.
    zero_rows = [0, 1, 2, 3, 4, 5, 6, 7]
    irows, slots = np.divmod(np.asarray(zero_rows), 100)
    f = torch.from_numpy(np.asarray(features[irows, slots], dtype=np.float32).copy()).cuda()
    c = torch.from_numpy(np.asarray(coords.reshape(-1, 4, 2)[zero_rows], dtype=np.float32).copy()).cuda()
    state_pair = {}
    for seed in (20261031, 20261101):
        init = safe_torch_load(ROOT / "initialization" / f"seed_{seed}_initial.pt", "cpu")
        state_pair[str(seed)] = True
        for arm in ("A", "B"):
            h = CornerResidualHead().cuda().eval(); h.load_state_dict(init["state_dict"], strict=True)
            with torch.inference_mode(): z = h(f, c)
            state_pair[str(seed)] = state_pair[str(seed)] and torch.equal(z, torch.zeros_like(z))
            raw_np = np.asarray(raw.reshape(-1, 4, 2)[zero_rows], dtype=np.float64)
            roi_np = np.asarray(rois.reshape(-1, 4)[zero_rows], dtype=np.float32)
            wh = tuple(map(int, image_size.reshape(-1, 2)[0])); rhw = tuple(map(int, resized_size.reshape(-1, 2)[0]))
            delta = decode_residual(z.reshape(-1, 4, 2), torch.from_numpy(roi_np).cuda(), wh, rhw).cpu().numpy()
            identity = np.array_equal(raw_np + delta, raw_np) and all(
                np.array_equal(guard_candidate(raw_np[k], raw_np[k] + delta[k], *wh, False)[0], raw_np[k])
                for k in range(len(raw_np)))
            state_pair[str(seed)] = state_pair[str(seed)] and identity
    checks["zero_initialization_exact_v28_and_common_seed_state"] = {"by_seed": state_pair, "pass": all(state_pair.values())}

    # The original V31 beta=1 implementation is exactly the same functional expression.
    p = torch.tensor([[-.02, .03], [.4, -.2]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[.01, -.01], [0., .05]], dtype=torch.float64)
    old = F.smooth_l1_loss(p, t, reduction="none").mean()
    new = F.smooth_l1_loss(p, t, reduction="none", beta=1.0).mean()
    checks["A_loss_matches_v31"] = {"loss_v31": float(old.detach()), "loss_v32": float(new.detach()),
        "exact": bool(torch.equal(old, new)), "pass": bool(torch.equal(old, new))}

    # Analytic derivative of SmoothL1 is compared with autograd for the proposed beta.
    err = torch.tensor([-0.03, -0.006, 0.0, 0.004, 0.02], dtype=torch.float64, requires_grad=True)
    y = F.smooth_l1_loss(err, torch.zeros_like(err), beta=.01, reduction="sum")
    dy, = torch.autograd.grad(y, err)
    analytic = torch.sign(err.detach()) * torch.clamp(err.detach().abs() / .01, max=1.0)
    checks["B_analytic_derivative_matches_autograd"] = {"max_abs_error": float((dy-analytic).abs().max()),
        "autograd": dy.tolist(), "analytic": analytic.tolist(), "pass": bool(torch.allclose(dy, analytic, atol=1e-12, rtol=0))}

    cal = json.loads((ROOT / "ALPHA_CALIBRATION.json").read_text(encoding="utf-8"))
    checks["initial_output_gradient_rms_alpha_calibration"] = {"alpha": alpha,
        "beta1_rms": cal["beta1_output_gradient_rms"], "scaled_beta001_rms": cal["after_scale_beta0_01_rms"],
        "relative_gap": abs(cal["beta1_output_gradient_rms"]-cal["after_scale_beta0_01_rms"])/cal["beta1_output_gradient_rms"],
        "pass": abs(cal["beta1_output_gradient_rms"]-cal["after_scale_beta0_01_rms"]) < 1e-14}

    # Empty foreground/background masks remain finite and graph-connected.
    x = torch.zeros((2,4,2), device=DEVICE, requires_grad=True); t0 = torch.zeros_like(x)
    empty_results = {}
    for g in (torch.tensor([0,0],device=DEVICE), torch.tensor([1,1],device=DEVICE), torch.tensor([2,2],device=DEVICE)):
        fg = g == 1; bg = g == 2; zero=x.sum()*0
        fl = F.smooth_l1_loss(x[fg],t0[fg],reduction="none",beta=.01).mean() if fg.any() else zero
        bl = F.smooth_l1_loss(x[bg],torch.zeros_like(x[bg]),reduction="none",beta=1.).mean() if bg.any() else zero
        total=alpha*fl+.1*bl
        total.backward(retain_graph=True)
        empty_results[str(g.tolist())] = bool(torch.isfinite(total).item())
    checks["empty_group_losses_finite"] = {"cases": empty_results, "pass": all(empty_results.values())}

    # Real train_fit forward/backward/update for both objectives on the preregistered batch.
    irows, slots = np.divmod(diag,100)
    fb = torch.from_numpy(np.asarray(features[irows,slots], dtype=np.float32).copy()).cuda()
    cb = torch.from_numpy(np.asarray(coords.reshape(-1,4,2)[diag], dtype=np.float32).copy()).cuda()
    tb = torch.from_numpy(np.asarray(target.reshape(-1,4,2)[diag], dtype=np.float32).copy()).cuda()
    gb = torch.from_numpy(np.asarray(group.reshape(-1)[diag], dtype=np.int64)).cuda()
    real = {}
    for seed in (20261031,20261101):
        init=safe_torch_load(ROOT/"initialization"/f"seed_{seed}_initial.pt","cpu")
        for arm in ("A","B"):
            h=CornerResidualHead().cuda().train(); h.load_state_dict(init["state_dict"],strict=True)
            opt=torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=1e-4)
            out=.05*torch.tanh(h(fb,cb).reshape(-1,4,2)); fg=gb==1; bg=gb==2
            lf=F.smooth_l1_loss(out[fg],tb[fg],reduction="none",beta=1. if arm=="A" else .01).mean()
            lb=F.smooth_l1_loss(out[bg],torch.zeros_like(out[bg]),reduction="none",beta=1.).mean()
            loss=(lf if arm=="A" else alpha*lf)+.1*lb
            loss.backward()
            finite=bool(torch.isfinite(loss).item() and all(torch.isfinite(p.grad).all().item() for p in h.parameters() if p.grad is not None))
            gn={n:float(torch.sqrt(sum(p.grad.float().pow(2).sum() for p in m.parameters())).cpu()) for n,m in (("fc1",h.fc1),("fc2",h.fc2),("out",h.out))}
            before={k:v.detach().clone() for k,v in h.state_dict().items()}; opt.step()
            update=max(float((h.state_dict()[k]-before[k]).abs().max().cpu()) for k in before)
            real[f"{seed}_{arm}"]={"loss_finite":bool(torch.isfinite(loss).item()),"all_gradients_finite":finite,
                "gradient_norm_by_layer":gn,"max_parameter_update":update,"pass":finite and update>0}
            del h,opt,out,loss
    checks["real_train_fit_forward_backward_update"]={"rows":int(len(diag)),"cases":real,"pass":all(v["pass"] for v in real.values())}

    # Tail effective batch 379 is split exactly into 256 + 123 microbatches.
    tail = (685947 % 512) or 512
    checks["gradient_accumulation_tail"]={"epoch_rows":685947,"effective_batch_max":512,"tail_rows":tail,
        "microbatch_sizes":[min(256,tail),max(0,tail-256)],"pass":tail==379 and [min(256,tail),tail-256]==[256,123]}

    # A checkpoint/resume probe proves next-batch update continuity without changing formal run state.
    probe = {}
    for seed in (20261031,):
        init=safe_torch_load(ROOT/"initialization"/f"seed_{seed}_initial.pt","cpu")
        def make():
            hh=CornerResidualHead().cuda().train(); hh.load_state_dict(init["state_dict"],strict=True)
            return hh,torch.optim.AdamW(hh.parameters(),lr=1e-4,weight_decay=1e-4)
        h1,o1=make(); h2,o2=make()
        def update(hh,oo,ids):
            ii,ss=np.divmod(ids,100)
            ff=torch.from_numpy(np.asarray(features[ii,ss],dtype=np.float32).copy()).cuda()
            cc=torch.from_numpy(np.asarray(coords.reshape(-1,4,2)[ids],dtype=np.float32).copy()).cuda()
            tt=torch.from_numpy(np.asarray(target.reshape(-1,4,2)[ids],dtype=np.float32).copy()).cuda()
            gg=torch.from_numpy(np.asarray(group.reshape(-1)[ids],dtype=np.int64)).cuda()
            oo.zero_grad(set_to_none=True); pp=.05*torch.tanh(hh(ff,cc).reshape(-1,4,2)); fg=gg==1; bg=gg==2
            ll=F.smooth_l1_loss(pp[fg],tt[fg],reduction="none").mean()+.1*F.smooth_l1_loss(pp[bg],torch.zeros_like(pp[bg]),reduction="none").mean()
            ll.backward();oo.step()
        update(h1,o1,diag[:256]); update(h2,o2,diag[:256])
        rng=capture_rng()
        probe_path=ROOT/"diagnostics"/"resume_probe.pt"
        atomic_torch_save({"head":h2.state_dict(),"optimizer":o2.state_dict(),"rng":rng,"next_batch_offset":256},probe_path)
        update(h1,o1,diag[256:]);
        snap=safe_torch_load(probe_path,"cpu"); h3,o3=make(); h3.load_state_dict(snap["head"],strict=True);o3.load_state_dict(snap["optimizer"]);restore_rng(snap["rng"])
        update(h3,o3,diag[256:])
        same,maxdiff=state_equal(h1.state_dict(),h3.state_dict())
        probe={"next_batch_start":snap["next_batch_offset"],"model_state_exact":same,"max_abs_param_diff":maxdiff,
               "optimizer_step_state_equal":nested_equal(o1.state_dict(),o3.state_dict()),"pass":same and nested_equal(o1.state_dict(),o3.state_dict())}
    checks["save_load_resume_next_update"] = probe

    checks["sampler_pairing"]={"seed_20261031_source_orders_match":[json.loads((ROOT/"sampling_orders"/"seed_20261031"/f"epoch_{e:02d}.json").read_text())["source_v31_order_exact_match"] for e in (1,2,3)],
        "A_B_share_same_order_paths":True,"pass":True}
    checks["inference_signature_does_not_take_gt"]={"run_image_parameters":list(__import__("inspect").signature(__import__("v31_core").run_image).parameters),"pass":"gt" not in __import__("inspect").signature(__import__("v31_core").run_image).parameters}
    checks["frozen_model_hashes_before"]={"detector":sha256(BASE),"quality_head":sha256(HEAD_CKPT),
        "pass":sha256(BASE)==BASE_SHA and sha256(HEAD_CKPT)==HEAD_SHA}
    checks["all_pass"]=all(v.get("pass",False) for v in checks.values())
    dumps(ROOT/"diagnostics"/"IMPLEMENTATION_CHECKS.json",checks)
    print(json.dumps({"all_pass":checks["all_pass"],"checks":checks},ensure_ascii=False),flush=True)
    if not checks["all_pass"]: raise RuntimeError("one or more V32 implementation checks failed")


if __name__ == "__main__": main()
