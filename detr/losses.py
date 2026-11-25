"""Set-prediction loss with auxiliary decoding losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from detr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from detr.matcher import HungarianMatcher


class SetCriterion(nn.Module):
    """Hungarian loss: weighted CE over all queries + L1 and GIoU over matched queries.

    Applied to the final decoder layer and (if ``aux_loss``) to every intermediate layer.
    Box losses are normalised by the number of ground-truth boxes in the batch. All terms
    are computed in float32, independent of the model's autocast dtype.
    """

    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_ce: float = 1.0,
                 weight_bbox: float = 5.0, weight_giou: float = 2.0, eos_coef: float = 0.1,
                 aux_loss: bool = True):
        super().__init__()
        self.num_classes, self.matcher, self.aux_loss = num_classes, matcher, aux_loss
        self.weights = {"loss_ce": weight_ce, "loss_bbox": weight_bbox, "loss_giou": weight_giou}
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _single_layer(self, out: dict, targets: list[dict], num_boxes: float) -> dict[str, Tensor]:
        indices = self.matcher(out, targets)
        logits = out["pred_logits"].float()
        dev = logits.device
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]).to(dev)
        src_idx = torch.cat([src for src, _ in indices]).to(dev)

        target_classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=dev)
        matched_labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes[batch_idx, src_idx] = matched_labels
        loss_ce = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)

        src_boxes = out["pred_boxes"][batch_idx, src_idx].float()
        tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes

        with torch.no_grad():
            if len(matched_labels):
                acc = (logits[batch_idx, src_idx].argmax(-1) == matched_labels).float().mean()
            else:
                acc = torch.tensor(1.0, device=dev)
        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou, "match_acc": acc}

    def _weighted(self, losses: dict[str, Tensor]) -> Tensor:
        return sum(w * losses[k] for k, w in self.weights.items())

    def forward(self, outputs: dict, targets: list[dict]) -> dict:
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        final = self._single_layer(outputs, targets, num_boxes)
        layer_losses = []
        if self.aux_loss:
            layer_losses = [self._weighted(self._single_layer(aux, targets, num_boxes))
                            for aux in outputs["aux_outputs"]]
        layer_losses.append(self._weighted(final))
        return {**final, "loss": sum(layer_losses), "layer_losses": [x.detach() for x in layer_losses]}
