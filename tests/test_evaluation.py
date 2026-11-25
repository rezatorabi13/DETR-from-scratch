import numpy as np
import pytest
import torch

from detr.evaluation import DetectionEvaluator


def _t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype)


GT = {"boxes": _t([[0, 0, 10, 10], [20, 20, 30, 30]]), "labels": _t([0, 0], torch.long),
      "difficult": _t([False, False], torch.bool)}


def _map50(pred, gt=GT):
    ev = DetectionEvaluator(1, iou_thresholds=np.array([0.5]))
    ev.update({"boxes": _t(pred["boxes"]), "scores": _t(pred["scores"]),
               "labels": _t(pred["labels"], torch.long)}, gt)
    return ev.compute()["mAP50"]


@pytest.mark.parametrize(
    ("pred", "expected"),
    [
        ({"boxes": [[0, 0, 10, 10], [20, 20, 30, 30]], "scores": [0.9, 0.8], "labels": [0, 0]}, 1.0),
        ({"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [0]}, 0.5),
        ({"boxes": [[50, 50, 60, 60], [0, 0, 10, 10], [20, 20, 30, 30]], "scores": [0.95, 0.9, 0.8],
          "labels": [0, 0, 0]}, 2 / 3),
        ({"boxes": [[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], "scores": [0.9, 0.85, 0.8],
          "labels": [0, 0, 0]}, 0.5 + 0.5 * 2 / 3),
    ],
    ids=["perfect", "half-recall", "false-positive-first", "duplicate"],
)
def test_average_precision_cases(pred, expected):
    assert _map50(pred) == pytest.approx(expected)


def test_difficult_objects_are_ignored():
    gt = {**GT, "difficult": _t([False, True], torch.bool)}
    assert _map50({"boxes": [[0, 0, 10, 10]], "scores": [0.9], "labels": [0]}, gt) == pytest.approx(1.0)


def test_coco_style_map_penalises_loose_boxes():
    ev = DetectionEvaluator(1)
    loose = {"boxes": _t([[0, 0, 12, 12]]), "scores": _t([0.9]), "labels": _t([0], torch.long)}
    ev.update(loose, {"boxes": _t([[0, 0, 10, 10]]), "labels": _t([0], torch.long),
                      "difficult": _t([False], torch.bool)})
    res = ev.compute()
    assert res["mAP50"] == pytest.approx(1.0)
    assert res["mAP"] < 1.0
