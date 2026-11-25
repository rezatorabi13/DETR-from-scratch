"""VOC-protocol mean Average Precision (all-point interpolation, 'difficult' objects ignored)."""

from __future__ import annotations

import numpy as np
from torch import Tensor

from detr.box_ops import box_iou

COCO_IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the monotone precision envelope (VOC 2010+ protocol)."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class DetectionEvaluator:
    """Accumulates per-image predictions / ground truth and computes AP per class and IoU.

    ``update(pred, gt)`` expects
      pred: ``boxes [N, 4]`` (xyxy), ``scores [N]``, ``labels [N]``
      gt:   ``boxes [M, 4]`` (xyxy), ``labels [M]``, ``difficult [M]`` (bool)
    in the same coordinate frame (e.g. original-image pixels).
    """

    def __init__(self, num_classes: int, iou_thresholds: np.ndarray = COCO_IOU_THRESHOLDS):
        self.num_classes = num_classes
        self.iou_thresholds = np.asarray(iou_thresholds)
        self.images: list[tuple[dict, dict]] = []

    def update(self, pred: dict[str, Tensor], gt: dict[str, Tensor]) -> None:
        self.images.append(({k: v.detach().cpu() for k, v in pred.items()},
                            {k: v.detach().cpu() for k, v in gt.items()}))

    def _match_class(self, c: int):
        detections, cache, n_gt = [], [], 0
        for i, (p, g) in enumerate(self.images):
            pm, gm = p["labels"] == c, g["labels"] == c
            pb, ps = p["boxes"][pm], p["scores"][pm]
            gb, gd = g["boxes"][gm], g["difficult"][gm]
            n_gt += int((~gd).sum())
            iou = box_iou(pb, gb)[0].numpy() if len(pb) and len(gb) else np.zeros((len(pb), len(gb)))
            cache.append((iou, gd.numpy()))
            detections += [(s, i, k) for k, s in enumerate(ps.tolist())]

        detections.sort(key=lambda d: -d[0])
        scores = np.array([d[0] for d in detections])
        tp = np.zeros((len(self.iou_thresholds), len(detections)))
        fp = np.zeros_like(tp)
        for t, thr in enumerate(self.iou_thresholds):
            used = [np.zeros(len(diff), dtype=bool) for _, diff in cache]
            for n, (_, i, k) in enumerate(detections):
                iou, diff = cache[i]
                if iou.shape[1] == 0:
                    fp[t, n] = 1
                    continue
                j = int(iou[k].argmax())
                if iou[k, j] < thr:
                    fp[t, n] = 1
                elif diff[j]:
                    continue  # matches a 'difficult' object: neither TP nor FP
                elif not used[i][j]:
                    tp[t, n], used[i][j] = 1, True
                else:
                    fp[t, n] = 1  # duplicate detection
        return tp, fp, scores, n_gt

    def compute(self) -> dict:
        ap = np.zeros((self.num_classes, len(self.iou_thresholds)))
        pr_curves = {}
        for c in range(self.num_classes):
            tp, fp, scores, n_gt = self._match_class(c)
            for t in range(len(self.iou_thresholds)):
                ctp, cfp = np.cumsum(tp[t]), np.cumsum(fp[t])
                recall = ctp / max(n_gt, 1)
                precision = ctp / np.maximum(ctp + cfp, 1e-9)
                ap[c, t] = average_precision(recall, precision) if n_gt else 0.0
                if t == 0:
                    pr_curves[c] = (recall, precision, scores)
        return {
            "mAP50": float(ap[:, 0].mean()),
            "mAP": float(ap.mean()),
            "AP50_per_class": ap[:, 0].tolist(),
            "AP_per_class": ap.mean(1).tolist(),
            "pr_curves": pr_curves,
        }
