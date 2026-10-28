from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(os.environ.get("DMKPD_V32_RUN_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
SOURCE_DATA_ROOT = Path(os.environ.get(
    "DMKPD_V32_SOURCE_ROOT", r"<PROJECT_ROOT>\DMKPD-V31-V28学习式角点残差"
)).resolve()
PROJECT = Path(r"<PROJECT_ROOT>")
V28 = PROJECT / "DMKPD-V28-V10亚像素解码-配对复核"
MANIFEST = PROJECT / "07_第5版_R6正式复跑_20261006" / "inputs" / "split_manifest_r2_internal.csv"
BASE = PROJECT / "DMKPD-V9-20261009" / "train" / "A_main" / "epoch_12.pt"
HEAD_CKPT = PROJECT / "DMKPD-V10-20261010" / "train" / "geometry_seed20261010" / "epoch_10.pt"
BASE_SHA = "89cfc3b044db37c4232cbc6f9e839b0b40bbe3fe2beb51a2e250cc3e47633a88"
HEAD_SHA = "5fa3bc7156c419799db5c7592955a28e05ca8602e9a4cfb7fdb2b1b755c67abf"
GRID = 4
MAX_CANDIDATES = 100
HIDDEN = 128
HIDDEN2 = 64
RESIDUAL_LIMIT = 0.05
SEED = 20261031
CODE_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = CODE_ROOT / "code" / "source_snapshot"
SOURCE_REFS = CODE_ROOT / "code" / "source_refs"
sys.path.insert(0, str(SNAPSHOT))
sys.path.insert(0, str(SOURCE_REFS))

import v10_rank as rank  # noqa: E402
import v9_evaluate as ev  # noqa: E402
import v9_geometry as geom  # noqa: E402
from run_v28_fixed import refined_decode  # noqa: E402


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_flat_candidate_rows(flat_rows: np.ndarray, n_images: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Decode image_index * MAX_CANDIDATES + candidate_slot sample IDs."""
    flat = np.asarray(flat_rows, dtype=np.int64)
    if np.any(flat < 0):
        raise IndexError("flattened candidate rows must be nonnegative")
    image_rows = flat // MAX_CANDIDATES
    candidate_slots = flat % MAX_CANDIDATES
    if n_images is not None and image_rows.size and int(image_rows.max()) >= int(n_images):
        raise IndexError(f"flattened candidate row exceeds image count {n_images}")
    return image_rows, candidate_slots


def dump_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_v28_backend() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_models() -> tuple[torch.nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
    if sha256(BASE) != BASE_SHA:
        raise RuntimeError(f"detector SHA256 mismatch: {BASE}")
    if sha256(HEAD_CKPT) != HEAD_SHA:
        raise RuntimeError(f"quality head SHA256 mismatch: {HEAD_CKPT}")
    set_v28_backend()
    model = rank.frozen_model()
    model.eval()
    model.requires_grad_(False)
    ckpt = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
    quality_head = rank.RankHead(int(ckpt["dim"])).cuda().eval()
    quality_head.load_state_dict(ckpt["model"], strict=True)
    quality_head.requires_grad_(False)
    mean = torch.as_tensor(ckpt["mean"], device="cuda", dtype=torch.float32)
    std = torch.as_tensor(ckpt["std"], device="cuda", dtype=torch.float32)
    if int(ckpt["dim"]) != 1083 or mean.numel() != 1083 or std.numel() != 1083:
        raise RuntimeError("V10 quality feature identity/dimension mismatch")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise RuntimeError("V10 quality normalization is invalid")
    inventory = rank.norm_inventory(model)
    if inventory["normalization_modules"] != {"FrozenBatchNorm2d": 53}:
        raise RuntimeError(f"FrozenBatchNorm inventory mismatch: {inventory['normalization_modules']}")
    if any(p.requires_grad for p in model.parameters()) or any(p.requires_grad for p in quality_head.parameters()):
        raise RuntimeError("a frozen detector/quality parameter is trainable")
    return model, quality_head, mean, std


def install_capture(model: torch.nn.Module) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def box_hook(_module, _inputs, output):
        captured["box_features"] = output.detach()

    def heat_hook(_module, _inputs, output):
        captured["heatmaps"] = output.detach()

    def roi_hook(_module, _inputs, output):
        captured["roi_features"] = output.detach()

    def roi_pre_hook(_module, args):
        # torchvision Keypoint RoI proposals are the post-regression boxes
        # that directly feed the four-corner pooling operation.
        captured["rois_resized"] = args[1][0].detach().clone()
        captured["image_shape_resized"] = tuple(int(x) for x in args[2][0])

    model.roi_heads.box_head.register_forward_hook(box_hook)
    model.roi_heads.keypoint_predictor.register_forward_hook(heat_hook)
    model.roi_heads.keypoint_roi_pool.register_forward_hook(roi_hook)
    model.roi_heads.keypoint_roi_pool.register_forward_pre_hook(roi_pre_hook)
    return captured


def roi_corner_coordinates(corners_original: np.ndarray, rois_resized: np.ndarray,
                           original_size: tuple[int, int], resized_size: tuple[int, int]) -> np.ndarray:
    """Encode raw keypoint-channel corners in the resized detector RoI frame."""
    width, height = map(float, original_size)
    resized_h, resized_w = map(float, resized_size)
    scale = np.asarray([resized_w / width, resized_h / height], dtype=np.float64)
    roi = np.asarray(rois_resized, dtype=np.float64)
    wh = np.maximum(roi[:, 2:4] - roi[:, 0:2], 1.0)
    resized_corners = np.asarray(corners_original, dtype=np.float64) * scale[None, None, :]
    normalized = (resized_corners - roi[:, None, 0:2]) / wh[:, None, :]
    return normalized.astype(np.float32)


def decode_residual(z: torch.Tensor, rois_resized: torch.Tensor | np.ndarray,
                    original_size: tuple[int, int], resized_size: tuple[int, int]) -> torch.Tensor:
    """Convert bounded ROI-relative offsets back to original-image pixels."""
    if z.ndim == 2:
        z = z.reshape(-1, 4, 2)
    roi = torch.as_tensor(rois_resized, device=z.device, dtype=z.dtype)
    roi_wh = (roi[:, 2:4] - roi[:, 0:2]).clamp_min(1.0)
    delta_resized = RESIDUAL_LIMIT * torch.tanh(z) * roi_wh[:, None, :]
    width, height = map(float, original_size)
    resized_h, resized_w = map(float, resized_size)
    scale_back = torch.as_tensor([width / resized_w, height / resized_h], device=z.device, dtype=z.dtype)
    return delta_resized * scale_back[None, None, :]


class CornerResidualHead(nn.Module):
    """Compact spatial RoI residual head; one raw-index-preserving 8-D output."""
    def __init__(self, channels: int = 256, grid: int = GRID, hidden: int = HIDDEN,
                 hidden2: int = HIDDEN2):
        super().__init__()
        self.channels = int(channels)
        self.grid = int(grid)
        in_features = self.channels * self.grid * self.grid + 8
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden2)
        self.out = nn.Linear(hidden2, 8)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, spatial_features: torch.Tensor, corner_norm: torch.Tensor) -> torch.Tensor:
        if spatial_features.ndim != 4 or spatial_features.shape[1] != self.channels:
            raise ValueError(f"unexpected RoI feature shape: {tuple(spatial_features.shape)}")
        if spatial_features.shape[-2:] != (self.grid, self.grid):
            spatial_features = F.adaptive_avg_pool2d(spatial_features, (self.grid, self.grid))
        x = torch.cat((spatial_features.flatten(1), corner_norm.reshape(-1, 8)), dim=1)
        return self.out(F.relu(self.fc2(F.relu(self.fc1(x)))))


def roi_spatial_features(roi_features: torch.Tensor) -> torch.Tensor:
    if roi_features.ndim != 4 or roi_features.shape[1] != 256:
        raise ValueError(f"expected actual keypoint RoI feature [N,256,H,W], got {tuple(roi_features.shape)}")
    return F.adaptive_avg_pool2d(roi_features.float(), (GRID, GRID))


def canonical_with_permutation(raw: np.ndarray, width: int, height: int):
    """Mirror V30 valid_quad: canonical point index -> original input index."""
    raw = np.asarray(raw, dtype=np.float64)
    quad, error = rank.parse_prediction(raw.tolist(), int(width), int(height))
    if quad is None or error:
        return quad, error or "invalid", None, False
    q = np.asarray(quad, dtype=np.float64)
    permutation: list[int] = []
    for point in q:
        distances = np.linalg.norm(raw - point[None, :], axis=1)
        order = np.argsort(distances, kind="stable")
        if (not len(order) or distances[order[0]] > 1e-8 or
                (len(order) > 1 and distances[order[1]] - distances[order[0]] <= 1e-10)):
            return quad, "", None, True
        permutation.append(int(order[0]))
    if len(permutation) != 4 or sorted(permutation) != [0, 1, 2, 3]:
        return quad, "", None, True
    return quad, "", tuple(permutation), True


def guard_candidate(original: np.ndarray, proposal: np.ndarray, width: int, height: int, triggered: bool):
    """Exact candidate-level V30 legality and raw-index permutation guard."""
    original = np.asarray(original, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if not triggered or np.array_equal(original, proposal):
        return original.copy(), "unchanged", ["no_coordinate_change"]
    _aq, _ae, ap, av = canonical_with_permutation(original, width, height)
    _bq, _be, bp, bv = canonical_with_permutation(proposal, width, height)
    if not av:
        tags = ["original_invalid"]
        if bv:
            tags.append("original_invalid_proposal_valid")
        return original.copy(), "reverted", tags
    if not bv:
        return original.copy(), "reverted", ["proposal_invalid"]
    if ap is None or bp is None:
        return original.copy(), "reverted", ["unreliable_permutation"]
    if ap != bp:
        return original.copy(), "reverted", ["point_order_changed"]
    return proposal.copy(), "accepted", ["accepted_same_permutation"]


def run_image(model, captured: dict[str, Any], quality_head: nn.Module, mean: torch.Tensor,
              std: torch.Tensor, image_path: str | Path) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_size = tuple(int(x) for x in image.size)
        tensor = to_tensor(image).cuda()
    with torch.inference_mode():
        model_start = torch.cuda.Event(enable_timing=True)
        model_end = torch.cuda.Event(enable_timing=True)
        model_start.record()
        prediction = model([tensor])[0]
        model_end.record()
        model_end.synchronize()
        model_gpu_ms = float(model_start.elapsed_time(model_end))
        ids = prediction["candidate_ids"].long()
        rois = captured["rois_resized"]
        roi_features = captured["roi_features"]
        if len(ids) != len(rois) or len(ids) != len(roi_features):
            raise RuntimeError(f"candidate/RoI order mismatch: {len(ids)}, {len(rois)}, {len(roi_features)}")
        # GeneralizedRCNN postprocess has already mapped prediction boxes back
        # to original-image coordinates. The keypoint RoI hook sees the same
        # regressed boxes in the transformed (resized) image frame.
        width, height = original_size
        resized_h, resized_w = captured["image_shape_resized"]
        resize_scale = prediction["boxes"].new_tensor(
            [resized_w / width, resized_h / height, resized_w / width, resized_h / height]
        )
        boxes_resized = prediction["boxes"].float() * resize_scale
        if not torch.allclose(boxes_resized, rois.float(), atol=2e-4, rtol=0):
            max_error = float((boxes_resized - rois.float()).abs().max().item())
            raise RuntimeError(f"post-regression box/RoI mapping mismatch after resize: {max_error}")
        if int(roi_features.shape[1]) != 256:
            raise RuntimeError(f"unexpected keypoint RoI channels {roi_features.shape[1]}")
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
        raw_native, refined, patches, states, offsets = refined_decode(
            captured["heatmaps"], rois, original_size, captured["image_shape_resized"])
        decode_end.record()
        decode_end.synchronize()
        decode_gpu_ms = float(decode_start.elapsed_time(decode_end))
        native_model = prediction["keypoints"][:, :, :2].detach().cpu().numpy()
        native_error = float(np.max(np.abs(native_model - raw_native))) if native_model.size else 0.0
        if native_error > 1e-4:
            raise RuntimeError(f"V28 native decode mismatch: {native_error}")
        raw_v28 = native_model.astype(np.float64) + (refined - raw_native)
        feature_start = torch.cuda.Event(enable_timing=True)
        feature_end = torch.cuda.Event(enable_timing=True)
        feature_start.record()
        raw_quality_features = rank.features(prediction, captured["heatmaps"], captured["box_features"])
        quantized_features = raw_quality_features.half().float()
        quality = quality_head((quantized_features - mean) / std, prediction["class_scores"].float())
        spatial = roi_spatial_features(roi_features)
        feature_end.record()
        feature_end.synchronize()
        feature_gpu_ms = float(feature_start.elapsed_time(feature_end))
        result = {
            "stem": Path(image_path).stem,
            "original_size": original_size,
            "resized_size": captured["image_shape_resized"],
            "candidate_ids": ids.detach().cpu().numpy().astype(np.int32),
            "class_scores": prediction["class_scores"].detach().cpu().numpy().astype(np.float32),
            "quality_scores": quality.detach().cpu().numpy().astype(np.float32),
            "boxes_original": prediction["boxes"].detach().cpu().numpy().astype(np.float32),
            "boxes_resized": boxes_resized.detach().cpu().numpy().astype(np.float32),
            "rois_resized": rois.detach().cpu().numpy().astype(np.float32),
            "raw_native": raw_native.astype(np.float64),
            "raw_v28": raw_v28.astype(np.float64),
            "raw_v28_online": raw_v28.astype(np.float64),
            "roi_corner_norm": roi_corner_coordinates(raw_v28, rois.detach().cpu().numpy(),
                                                       original_size, captured["image_shape_resized"]),
            "spatial_features": spatial.detach().cpu().numpy().astype(np.float32),
            "fit_state": states.astype(np.int8),
            "fit_offset_resized": offsets.astype(np.float64),
            "fit_neighborhood": patches.astype(np.float32),
            "native_decode_max_abs_error": native_error,
            "model_gpu_ms": model_gpu_ms,
            "decoder_gpu_ms": decode_gpu_ms,
            "quality_and_spatial_gpu_ms": feature_gpu_ms,
        }
    return result


def safe_torch_load(path: str | Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def prediction_from_pool(pool: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return ev.rerank_and_nms(pool, "q", 0.30, 100)


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
