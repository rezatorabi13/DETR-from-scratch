"""Hungarian matching between predictions and ground truth."""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from detr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """One-to-one assignment minimising ``w_cls * -p(class) + w_L1 * L1 + w_giou * -GIoU``.

    The cost matrix is computed for the whole batch at once on the device; the assignment
    (``scipy.optimize.linear_sum_assignment``) is solved per image on the CPU.
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> list[tuple[Tensor, Tensor]]:
        B, Q = outputs["pred_logits"].shape[:2]
        sizes = [len(t["boxes"]) for t in targets]
        empty = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        if sum(sizes) == 0:
            return [empty for _ in range(B)]

        prob = outputs["pred_logits"].flatten(0, 1).float().softmax(-1)
        out_box = outputs["pred_boxes"].flatten(0, 1).float()
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_box = torch.cat([t["boxes"] for t in targets])

        cost = (self.cost_bbox * torch.cdist(out_box, tgt_box, p=1)
                - self.cost_class * prob[:, tgt_ids]
                - self.cost_giou * generalized_box_iou(box_cxcywh_to_xyxy(out_box),
                                                       box_cxcywh_to_xyxy(tgt_box)))
        cost = cost.view(B, Q, -1).cpu()

        indices = []
        for i, c in enumerate(cost.split(sizes, dim=-1)):
            if sizes[i] == 0:
                indices.append(empty)
                continue
            rows, cols = linear_sum_assignment(c[i].numpy())
            indices.append((torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)))
        return indices
