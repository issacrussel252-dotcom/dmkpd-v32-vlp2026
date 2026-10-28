from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from v31_core import (MANIFEST, ROOT, CornerResidualHead, append_jsonl,
                      dump_json, load_models, run_image, set_v28_backend)
from v10_rank import read_rows


def main():
    set_v28_backend()
    rows = read_rows(MANIFEST, "train_fit")
    rows = sorted(rows, key=lambda r: hashlib.sha256(r["stem"].encode()).hexdigest())[:16]
    model, quality, mean, std = load_models()
    from v31_core import install_capture
    cap = install_capture(model)
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    with Image.open(rows[0]["image"]) as im:
        warm = to_tensor(im.convert("RGB")).cuda()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        e0.record()
        for _ in range(5):
            model([warm])
        e1.record(); e1.synchronize()
    warmup_seconds = float(e0.elapsed_time(e1)) / 1000.0
    samples = []
    for row in rows:
        t0 = time.perf_counter()
        result = run_image(model, cap, quality, mean, std, row["image"])
        samples.append({"stem": row["stem"], "wall_seconds": time.perf_counter()-t0,
                        "model_gpu_seconds": result["model_gpu_ms"]/1000,
                        "decode_gpu_seconds": result["decoder_gpu_ms"]/1000,
                        "quality_and_spatial_gpu_seconds": result["quality_and_spatial_gpu_ms"]/1000,
                        "candidate_count": len(result["candidate_ids"]),
                        "feature_shape": list(result["spatial_features"].shape)})
    gpu_per_image = [x["model_gpu_seconds"]+x["decode_gpu_seconds"]+x["quality_and_spatial_gpu_seconds"] for x in samples]
    sorted_gpu = sorted(gpu_per_image)
    median_gpu = statistics.median(gpu_per_image)
    p95_gpu = float(np.percentile(gpu_per_image, 95))
    head = CornerResidualHead().cuda().train()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-4)
    x = torch.randn(512, 256, 4, 4, device="cuda")
    c = torch.randn(512, 4, 2, device="cuda")
    t = torch.randn(512, 4, 2, device="cuda") * 0.01
    g = torch.tensor(([1, 2] * 256), device="cuda")
    step_times = []
    for _ in range(3):
        z = head(x, c).reshape(-1, 4, 2); pred = .05 * torch.tanh(z)
        loss = torch.nn.functional.smooth_l1_loss(pred, t)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        z = head(x, c).reshape(-1, 4, 2); pred = .05 * torch.tanh(z)
        fg = g == 1; bg = g == 2
        lf = torch.nn.functional.smooth_l1_loss(pred[fg], t[fg]) if fg.any() else pred.sum()*0
        lb = torch.nn.functional.smooth_l1_loss(pred[bg], torch.zeros_like(pred[bg])) if bg.any() else pred.sum()*0
        (lf + .1*lb).backward(); opt.step()
        end.record(); end.synchronize()
        step_times.append(float(start.elapsed_time(end))/1000)
    head.eval()
    with torch.inference_mode():
        eval_start = torch.cuda.Event(enable_timing=True); eval_end = torch.cuda.Event(enable_timing=True)
        eval_start.record()
        for _ in range(10):
            head(x[:100], c[:100])
        eval_end.record(); eval_end.synchronize()
    eval_100_gpu_seconds = float(eval_start.elapsed_time(eval_end))/1000/10
    train_images = 7731
    dev_images = 859
    max_selected_per_epoch = train_images * 100
    worst_train_updates = int(np.ceil(max_selected_per_epoch / 512))
    head_update_p95 = float(np.percentile(step_times, 95))
    extract_center = (train_images + dev_images) * median_gpu
    train_head_center = worst_train_updates * 3 * statistics.median(step_times)
    dev_head_eval_center = dev_images * 100 * 3 / 512 * statistics.median(step_times)
    online_confirm_center = dev_images * median_gpu
    # The conditional online pass is budgeted in advance even if the fixed gate
    # later fails; this keeps the resource decision independent of dev metrics.
    center = warmup_seconds + extract_center + train_head_center + dev_head_eval_center + online_confirm_center
    conservative = warmup_seconds + (train_images + dev_images) * max(p95_gpu, median_gpu * 1.5) + \
        worst_train_updates * 3 * max(head_update_p95, statistics.median(step_times) * 1.5) + \
        dev_images * 100 * 3 / 512 * max(head_update_p95, statistics.median(step_times) * 1.5) + \
        dev_images * max(p95_gpu, median_gpu * 1.5)
    result = {
        "status": "preflight_complete", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pilot_role": "train_fit image-only forward; no annotations or dev labels read",
        "pilot_stems": [r["stem"] for r in rows], "pilot": samples,
        "warmup_gpu_seconds": warmup_seconds,
        "median_gpu_seconds_per_image": median_gpu, "p95_gpu_seconds_per_image": p95_gpu,
        "head_train_step_seconds_batch512": {"median": statistics.median(step_times), "p95": head_update_p95,
                                              "samples": step_times},
        "head_eval_seconds_per_100_candidates": eval_100_gpu_seconds,
        "cache_bytes_estimate_train_fit_fp32": train_images*100*256*4*4*4,
        "cache_bytes_estimate_dev_fp32": dev_images*100*256*4*4*4,
        "worst_case_training_selected_candidates_per_epoch": max_selected_per_epoch,
        "estimated_worst_case_updates_per_epoch": worst_train_updates,
        "GPU_budget_hours": 8,
        "estimated_GPU_active_hours_center": center/3600,
        "estimated_GPU_active_hours_conservative": conservative/3600,
        "conservative_buffer_factor_for_per_image_compute": 1.5,
        "within_budget": conservative <= 8*3600,
        "cuda_device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "test300_read": False,
    }
    dump_json(ROOT / "diagnostics" / "RESOURCE_PREFLIGHT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["within_budget"]:
        raise RuntimeError("conservative GPU active estimate exceeds the fixed 8-hour budget")


if __name__ == "__main__":
    main()
