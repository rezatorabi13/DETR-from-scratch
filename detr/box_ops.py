"""Bounding-box utilities. Boxes are ``[cx, cy, w, h]`` (cxcywh) or ``[x1, y1, x2, y2]`` (xyxy)."""

from __future__ import annotations

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """Pairwise IoU of xyxy boxes ``[N, 4] x [M, 4]``. Returns ``(iou, union)``, each ``[N, M]``."""
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - inter
    return inter / union.clamp(min=1e-7), union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise Generalized IoU (Rezatofighi et al., 2019) of xyxy boxes, in ``[-1, 1]``."""
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (rb - lt).clamp(min=0).prod(-1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)
