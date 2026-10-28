"""DMKPD V9 Keypoint R-CNN with explicit FrozenBN and conditional quality."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    fastrcnn_loss,
    keypointrcnn_inference,
    keypointrcnn_loss,
)
from torchvision.ops import boxes as box_ops
from torchvision.ops.misc import FrozenBatchNorm2d

from v9_geometry import canonical_quad, local_oks, ordered_quad_is_valid, parse_shapes, polygon_area


V9_KIND_A = "dmkpd_v9_class_keypointrcnn"
V9_KIND_B = "dmkpd_v9_conditional_quality_keypointrcnn"


def build_model(
    quality: bool,
    candidate_budget: int = 300,
    pretrained_backbone: bool = False,
    score_formula: str | None = None,
    pool_formula: str = "p_cls",
) -> torch.nn.Module:
    """Always construct FrozenBatchNorm2d independent of weight-loading path."""
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=ResNet50_Weights.DEFAULT if pretrained_backbone else None,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=5,
    )
    model = KeypointRCNN(
        backbone,
        num_classes=2,
        num_keypoints=4,
        min_size=768,
        max_size=1024,
        box_score_thresh=0.0,
        box_nms_thresh=1.0,
        box_detections_per_img=int(candidate_budget),
    )
    roi = model.roi_heads
    roi.__class__ = V9RoIHeads
    # Freeze the actual torchvision RoI matcher policy explicitly. The RPN has
    # its own matcher and is intentionally untouched.
    roi.proposal_matcher.allow_low_quality_matches = False
    roi.quality_enabled = bool(quality)
    roi.score_formula = score_formula or ("p_cls*q" if quality else "p_cls")
    roi.pool_formula = pool_formula
    roi.last_quality_stats = {}
    roi.last_debug = []
    if quality:
        feature_dim = int(roi.box_head.fc7.out_features)
        roi.quality_head = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Linear(256, 1))
        nn.init.normal_(roi.quality_head[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(roi.quality_head[-1].bias, 0.0)
    return model


def norm_inventory(model: torch.nn.Module) -> dict[str, Any]:
    module_types: dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        if "BatchNorm" in name:
            module_types[name] = module_types.get(name, 0) + 1
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    return {
        "normalization_modules": module_types,
        "normalization_type": "FrozenBatchNorm2d",
        "trainable_parameter_names": trainable,
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().cpu()),
        "p10": float(quantiles[0].cpu()),
        "p50": float(quantiles[1].cpu()),
        "p90": float(quantiles[2].cpu()),
    }


class V9RoIHeads(RoIHeads):
    """Same class feature for class score/q; q estimates foreground-conditional quad OKS."""

    quality_enabled: bool
    score_formula: str
    pool_formula: str

    def _formula(self, p_cls: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        if self.score_formula == "p_cls":
            return p_cls
        if self.score_formula == "q":
            return quality
        if self.score_formula == "p_cls*q":
            return p_cls * quality
        if self.score_formula == "p_cls*sqrt(q)":
            return p_cls * torch.sqrt(quality.clamp_min(0.0))
        raise ValueError(f"unknown score formula: {self.score_formula}")

    @staticmethod
    def _scale_tags(target: dict[str, torch.Tensor], matched: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        small = target.get("gt_small")
        tiny = target.get("gt_tiny")
        if small is None or tiny is None:
            zeros = torch.zeros_like(matched, dtype=torch.bool)
            return zeros, zeros
        return small[matched].bool(), tiny[matched].bool()

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if self.training:
            if targets is None:
                raise ValueError("training requires train_fit targets")
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            matched_idxs = labels = regression_targets = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)
        quality_logits = self.quality_head(box_features).squeeze(1).float() if self.quality_enabled else None

        result: list[dict[str, torch.Tensor]] = []
        losses: dict[str, torch.Tensor] = {}
        if self.training:
            assert labels is not None and regression_targets is not None and matched_idxs is not None and targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes_per_image = [len(items) for items in proposals]
            decoded = self.box_coder.decode(box_regression, proposals)
            decoded_parts = decoded.split(boxes_per_image, dim=0)
            logits_parts = class_logits.split(boxes_per_image, dim=0)
            quality_parts = (
                torch.sigmoid(quality_logits).split(boxes_per_image, dim=0)
                if quality_logits is not None
                else [None] * len(boxes_per_image)
            )
            self.last_debug = []
            for image_index, (boxes_all, logits, quality, shape, rpn_proposals) in enumerate(
                zip(decoded_parts, logits_parts, quality_parts, image_shapes, proposals)
            ):
                p_all = F.softmax(logits.float(), dim=-1)[:, 1]
                q_all = quality if quality is not None else torch.ones_like(p_all)
                class_boxes = box_ops.clip_boxes_to_image(boxes_all.reshape(-1, 2, 4)[:, 1], shape)
                raw_indices = torch.arange(len(class_boxes), device=class_boxes.device)
                score_mask = p_all > self.score_thresh
                score_indices = raw_indices[score_mask]
                sized_mask = box_ops.remove_small_boxes(class_boxes[score_indices], min_size=1e-2)
                sized_indices = score_indices[sized_mask]
                if self.pool_formula == "p_cls":
                    pool_rank_score = p_all
                elif self.pool_formula == "q":
                    pool_rank_score = q_all
                elif self.pool_formula == "p_cls*q":
                    pool_rank_score = p_all * q_all
                elif self.pool_formula == "p_cls*sqrt(q)":
                    pool_rank_score = p_all * torch.sqrt(q_all.clamp_min(0.0))
                else:
                    raise ValueError(f"unknown pool formula: {self.pool_formula}")
                # Box NMS and pre-keypoint truncation depend only on the fixed
                # pool formula, never on the score ablation being measured.
                pre_nms = sized_indices[torch.argsort(pool_rank_score[sized_indices], descending=True, stable=True)]
                nms_indices_local = box_ops.batched_nms(
                    class_boxes[pre_nms], pool_rank_score[pre_nms], torch.ones_like(pre_nms), self.nms_thresh
                )
                nms_indices = pre_nms[nms_indices_local]
                nms_membership = torch.zeros_like(p_all, dtype=torch.bool)
                nms_membership[nms_indices] = True
                ranked_sized = sized_indices[torch.argsort(pool_rank_score[sized_indices], descending=True, stable=True)]
                class_pool_indices = ranked_sized[nms_membership[ranked_sized]]
                # V9 fixes the pre-keypoint pool by classifier rank. This keeps all
                # post-keypoint score formulas on exactly the same candidate set.
                pool_indices = class_pool_indices[: self.detections_per_img]
                nms_pool_indices = pool_indices
                keypoint_indices = pool_indices
                keypoint_boxes = class_boxes[keypoint_indices]
                selected_p = p_all[keypoint_indices]
                selected_q = q_all[keypoint_indices]
                selected_score = self._formula(selected_p, selected_q)
                result.append({
                    "boxes": keypoint_boxes,
                    "scores": selected_score,
                    "labels": torch.ones_like(keypoint_indices, dtype=torch.int64),
                    "class_scores": selected_p,
                    "quality_scores": selected_q,
                    "candidate_ids": keypoint_indices,
                })
                # Keep full pre-cap rows as compact tensors for per-stage recall and
                # explicit truncation attribution. Keypoints exist for pool_indices.
                self.last_debug.append({
                    "rpn_proposals": rpn_proposals.detach(),
                    "boxes_all": class_boxes.detach(),
                    "p_all": p_all.detach(),
                    "q_all": q_all.detach(),
                    "candidate_indices_after_score": score_indices.detach(),
                    "candidate_indices_after_size": sized_indices.detach(),
                    "candidate_indices_after_box_nms": nms_indices.detach(),
                    "candidate_indices_pool": keypoint_indices.detach(),
                    "candidate_indices_final_rank": nms_pool_indices.detach(),
                    "candidate_ids": keypoint_indices.detach(),
                    "image_shape_resized": shape,
                    "score_floor_removed": int((~score_mask).sum()),
                    "small_box_removed": int(len(score_indices) - len(sized_indices)),
                    "box_nms_removed": int(len(sized_indices) - len(nms_indices)),
                    "candidate_cap_removed": int(max(0, len(nms_indices) - self.detections_per_img)),
                })

        if self.keypoint_roi_pool is None or self.keypoint_head is None or self.keypoint_predictor is None:
            raise RuntimeError("four-corner Keypoint R-CNN head is required")

        if self.training:
            assert matched_idxs is not None and labels is not None and targets is not None
            positive_indices = [torch.where(image_labels > 0)[0] for image_labels in labels]
            keypoint_proposals = [image_proposals[pos] for image_proposals, pos in zip(proposals, positive_indices)]
            pos_matched_idxs = [image_matches[pos] for image_matches, pos in zip(matched_idxs, positive_indices)]
        else:
            keypoint_proposals = [prediction["boxes"] for prediction in result]
            pos_matched_idxs = []

        keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
        keypoint_features = self.keypoint_head(keypoint_features)
        keypoint_logits = self.keypoint_predictor(keypoint_features)

        if self.training:
            assert targets is not None and labels is not None and matched_idxs is not None and regression_targets is not None
            gt_keypoints = [target["keypoints"] for target in targets]
            losses["loss_keypoint"] = keypointrcnn_loss(keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs)
            if self.quality_enabled:
                assert quality_logits is not None
                # Quality targets are generated on the class-1 regressed/clipped
                # boxes, using the actual predicted corners and evaluator geometry.
                boxes_per_image = [len(items) for items in proposals]
                with torch.no_grad():
                    decoded = self.box_coder.decode(box_regression.detach(), proposals).reshape(-1, 2, 4)[:, 1]
                    decoded_parts = decoded.split(boxes_per_image, dim=0)
                    predicted_boxes: list[torch.Tensor] = []
                    for boxes, shape, pos in zip(decoded_parts, image_shapes, positive_indices):
                        predicted_boxes.append(box_ops.clip_boxes_to_image(boxes[pos], shape))
                    quality_kpt_features = self.keypoint_roi_pool(features, predicted_boxes, image_shapes)
                    quality_kpt_features = self.keypoint_head(quality_kpt_features)
                    quality_kpt_logits = self.keypoint_predictor(quality_kpt_features)
                    predicted_points, _ = keypointrcnn_inference(quality_kpt_logits, predicted_boxes)
                    targets_per_image: list[torch.Tensor] = []
                    scale_batches: dict[str, dict[str, list[float]]] = {
                        key: {"target": [], "prediction": []} for key in ("all", "small", "tiny", "regular")
                    }
                    invalid_targets = 0
                    pred_parts = quality_logits.detach().split(boxes_per_image, dim=0)
                    pcls_parts = F.softmax(class_logits.detach().float(), dim=-1).split(boxes_per_image, dim=0)
                    for image_index, (image_labels, pos, pos_matches, pred_pts, pred_boxes, target, qlogits, pcls) in enumerate(
                        zip(labels, positive_indices, pos_matched_idxs, predicted_points, predicted_boxes, targets, pred_parts, pcls_parts)
                    ):
                        values = torch.zeros((len(image_labels),), device=quality_logits.device, dtype=torch.float32)
                        if len(pos):
                            gt_pts = target["keypoints"][pos_matches, :, :2].detach().cpu().float().numpy()
                            pred_np = pred_pts[..., :2].detach().cpu().float().numpy()
                            h, w = image_shapes[image_index]
                            gt_small = target.get("gt_small", torch.zeros((len(target["boxes"]),), dtype=torch.bool))[pos_matches].detach().cpu().tolist()
                            gt_tiny = target.get("gt_tiny", torch.zeros((len(target["boxes"]),), dtype=torch.bool))[pos_matches].detach().cpu().tolist()
                            scores = torch.sigmoid(qlogits[pos]).detach().cpu().tolist()
                            for local_i, (points, gt, is_small, is_tiny) in enumerate(zip(pred_np, gt_pts, gt_small, gt_tiny)):
                                target_quality = 0.0
                                try:
                                    quad = canonical_quad(points.tolist())
                                    in_bounds = all(0.0 <= x < w and 0.0 <= y < h for x, y in quad)
                                    if ordered_quad_is_valid(quad) and in_bounds:
                                        gt_quad = canonical_quad(gt.tolist())
                                        target_quality = local_oks(quad, gt_quad)
                                    else:
                                        invalid_targets += 1
                                except (TypeError, ValueError, IndexError, OverflowError):
                                    invalid_targets += 1
                                values[pos[local_i]] = target_quality
                                target_np = float(target_quality)
                                group_names = ["all", "small" if is_small else "regular"]
                                if is_tiny:
                                    group_names.append("tiny")
                                for group in group_names:
                                    scale_batches[group]["target"].append(target_np)
                                    scale_batches[group]["prediction"].append(float(scores[local_i]))
                        targets_per_image.append(values)
                    quality_target = torch.cat(targets_per_image, dim=0)
                positive_mask = torch.cat(labels, dim=0) > 0
                positive_prob = torch.sigmoid(quality_logits.float())[positive_mask]
                positive_target = quality_target[positive_mask]
                if positive_prob.numel():
                    loss_quality_positive = F.smooth_l1_loss(positive_prob, positive_target, reduction="mean")
                else:
                    loss_quality_positive = quality_logits.sum() * 0.0
                # Conditional quality ignores background RoIs by design. The
                # negative quality loss and its gradient are exactly zero; class
                # loss alone models foreground probability.
                quality_loss = loss_quality_positive
                losses["loss_quality"] = quality_loss
                negative_mask = ~positive_mask
                distribution = {}
                for group, values in scale_batches.items():
                    if values["target"]:
                        t = torch.tensor(values["target"], dtype=torch.float32)
                        q = torch.tensor(values["prediction"], dtype=torch.float32)
                        distribution[group] = {"n": len(t), "target": _summary(t), "predicted_q": _summary(q)}
                    else:
                        distribution[group] = {"n": 0, "target": _summary(torch.empty(0)), "predicted_q": _summary(torch.empty(0))}
                neg_q = torch.sigmoid(quality_logits.detach().float()[negative_mask])
                positive_p = F.softmax(class_logits.detach().float(), dim=-1)[:, 1][positive_mask]
                negative_p = F.softmax(class_logits.detach().float(), dim=-1)[:, 1][negative_mask]
                self.last_quality_stats = {
                    "positive_rois": int(positive_mask.sum()),
                    "negative_rois": int(negative_mask.sum()),
                    "negative_quality_loss": 0.0,
                    "negative_quality_gradient_norm": 0.0,
                    "positive_quality_loss": float(loss_quality_positive.detach().cpu()),
                    "positive_q_prediction": _summary(positive_prob.detach()),
                    "positive_q_target": _summary(positive_target.detach()),
                    "negative_q_prediction": _summary(neg_q),
                    "positive_class_probability": _summary(positive_p),
                    "negative_class_probability": _summary(negative_p),
                    "invalid_or_out_of_bounds_positive_targets": invalid_targets,
                    "by_scale": distribution,
                }
            else:
                self.last_quality_stats = {}
        else:
            keypoints_prob, keypoint_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
            for points, point_scores, prediction in zip(keypoints_prob, keypoint_scores, result):
                prediction["keypoints"] = points
                prediction["keypoints_scores"] = point_scores
            for debug, pred in zip(self.last_debug, result):
                debug["keypoints"] = pred["keypoints"].detach()
                debug["boxes_selected"] = pred["boxes"].detach()

        return result, losses


def parse_prediction(points: list[list[float]], width: int, height: int) -> tuple[list[tuple[float, float]] | None, str]:
    try:
        quad = canonical_quad(points)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        return None, f"canonicalize:{type(exc).__name__}"
    if not ordered_quad_is_valid(quad):
        return quad, "ordered_geometry_invalid"
    if not all(0.0 <= x < width and 0.0 <= y < height for x, y in quad):
        return quad, "out_of_bounds"
    return quad, ""
