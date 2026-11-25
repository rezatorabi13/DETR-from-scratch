import torch
from torchvision import ops

from detr.box_ops import box_cxcywh_to_xyxy, box_iou, box_xyxy_to_cxcywh, generalized_box_iou


def _random_boxes(n, seed):
    g = torch.Generator().manual_seed(seed)
    return box_cxcywh_to_xyxy(torch.rand(n, 4, generator=g) * 0.5 + 0.1)


def test_format_roundtrip():
    b = _random_boxes(20, 0)
    assert torch.allclose(box_cxcywh_to_xyxy(box_xyxy_to_cxcywh(b)), b, atol=1e-6)
    assert torch.allclose(b, ops.box_convert(box_xyxy_to_cxcywh(b), "cxcywh", "xyxy"), atol=1e-6)


def test_iou_matches_torchvision():
    b1, b2 = _random_boxes(50, 1), _random_boxes(30, 2)
    assert torch.allclose(box_iou(b1, b2)[0], ops.box_iou(b1, b2), atol=1e-5)


def test_giou_matches_torchvision():
    b1, b2 = _random_boxes(50, 3), _random_boxes(30, 4)
    assert torch.allclose(generalized_box_iou(b1, b2), ops.generalized_box_iou(b1, b2), atol=1e-5)


def test_giou_identity_and_range():
    b = _random_boxes(10, 5)
    g = generalized_box_iou(b, b)
    assert torch.allclose(g.diag(), torch.ones(10), atol=1e-5)
    assert g.min() >= -1 and g.max() <= 1 + 1e-6
