import random

import torch
from PIL import Image

from detr.data import collate_fn
from detr.data.transforms import (
    RandomHorizontalFlip,
    RandomResize,
    RandomSizeCrop,
    ToTensorNormalize,
    build_transforms,
)


def _sample(w=200, h=100):
    img = Image.new("RGB", (w, h))
    tgt = {"boxes": torch.tensor([[10.0, 20.0, 60.0, 80.0]]), "labels": torch.tensor([1]),
           "difficult": torch.tensor([False])}
    return img, tgt


def test_flip_mirrors_boxes():
    img, tgt = _sample()
    _, out = RandomHorizontalFlip(p=1.0)(img, tgt)
    assert out["boxes"].tolist() == [[140.0, 20.0, 190.0, 80.0]]


def test_resize_scales_boxes_and_respects_max_size():
    img, tgt = _sample(200, 100)
    out_img, out = RandomResize([150], max_size=250)(img, tgt)
    assert out_img.size == (250, 125)  # long side capped
    assert torch.allclose(out["boxes"], tgt["boxes"] * 1.25)


def test_crop_keeps_boxes_inside_image():
    random.seed(0)
    img, tgt = _sample(400, 300)
    for _ in range(20):
        out_img, out = RandomSizeCrop(100, 200)(img, tgt)
        b = out["boxes"]
        assert len(b) == len(out["labels"]) == len(out["difficult"])
        assert (b[:, 0::2] >= 0).all() and (b[:, 0::2] <= out_img.width).all()
        assert (b[:, 1::2] >= 0).all() and (b[:, 1::2] <= out_img.height).all()


def test_to_tensor_outputs_normalised_cxcywh():
    img, tgt = _sample(200, 100)
    x, out = ToTensorNormalize()(img, tgt)
    assert x.shape == (3, 100, 200)
    assert torch.allclose(out["boxes"], torch.tensor([[0.175, 0.5, 0.25, 0.6]]))


def test_train_pipeline_and_padded_collate():
    random.seed(0)
    tf = build_transforms(True, [128, 160], max_size=256, eval_size=160)
    batch = [tf(*_sample(200, 100)), tf(*_sample(120, 180))]
    images, mask, targets = collate_fn(batch)
    assert images.shape[-2] % 32 == 0 and images.shape[-1] % 32 == 0
    for k, (x, _) in enumerate(batch):
        assert not mask[k, : x.shape[1], : x.shape[2]].any()
        assert mask[k].sum() == mask.shape[1] * mask.shape[2] - x.shape[1] * x.shape[2]
    assert all(((t["boxes"] >= 0) & (t["boxes"] <= 1)).all() for t in targets)
