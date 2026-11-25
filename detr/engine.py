"""Training and evaluation loops."""

from __future__ import annotations

import contextlib
from collections import defaultdict

import torch
from torch import Tensor, nn

from detr.box_ops import box_cxcywh_to_xyxy
from detr.evaluation import DetectionEvaluator


def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]


@torch.no_grad()
def postprocess(outputs: dict) -> list[dict[str, Tensor]]:
    """Model outputs -> per-image ``boxes`` (normalised xyxy), ``scores``, ``labels``. No NMS."""
    prob = outputs["pred_logits"].float().softmax(-1)[..., :-1]
    scores, labels = prob.max(-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"].float())
    return [{"boxes": b, "scores": s, "labels": lb} for b, s, lb in zip(boxes, scores, labels)]


def scale_boxes(boxes_xyxy: Tensor, size_hw: Tensor) -> Tensor:
    h, w = size_hw.tolist()
    return boxes_xyxy * torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)


def train_one_epoch(model: nn.Module, criterion: nn.Module, loader, optimizer, scheduler,
                    device: torch.device, amp: bool = True, clip_norm: float = 0.1,
                    accum_steps: int = 1) -> dict[str, float]:
    model.train()
    criterion.train()
    sums, n = defaultdict(float), 0
    optimizer.zero_grad(set_to_none=True)
    for it, (images, mask, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = to_device(targets, device)

        with autocast(device, amp):
            outputs = model(images, mask)
        losses = criterion(outputs, targets)
        (losses["loss"] / accum_steps).backward()

        if (it + 1) % accum_steps == 0:
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        for k in ("loss", "loss_ce", "loss_bbox", "loss_giou", "match_acc"):
            sums[k] += losses[k].item()
        for i, layer_loss in enumerate(losses["layer_losses"]):
            sums[f"layer{i + 1}"] += layer_loss.item()
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int,
             amp: bool = True, return_predictions: bool = False):
    """Run the model on ``loader`` and compute VOC mAP in original-image pixel coordinates."""
    model.eval()
    evaluator = DetectionEvaluator(num_classes)
    predictions = []
    for images, mask, targets in loader:
        with autocast(device, amp):
            outputs = model(images.to(device, non_blocking=True), mask.to(device, non_blocking=True))
        for pred, t in zip(postprocess(outputs), targets):
            pred = {k: v.cpu() for k, v in pred.items()}
            pred["boxes"] = scale_boxes(pred["boxes"], t["orig_size"])
            gt = {"boxes": scale_boxes(box_cxcywh_to_xyxy(t["boxes"]), t["orig_size"]),
                  "labels": t["labels"], "difficult": t["difficult"]}
            evaluator.update(pred, gt)
            if return_predictions:
                predictions.append((pred, gt))
    results = evaluator.compute()
    return (results, predictions) if return_predictions else results
