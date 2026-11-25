"""Box-aware augmentations.

Every transform maps ``(PIL image, target) -> (PIL image, target)``. Until the final
:class:`ToTensorNormalize`, ``target["boxes"]`` holds absolute ``xyxy`` pixel coordinates;
afterwards it holds normalised ``cxcywh`` boxes relative to the (unpadded) image.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tgt):
        for t in self.transforms:
            img, tgt = t(img, tgt)
        return img, tgt


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, tgt):
        if random.random() < self.p:
            img = TF.hflip(img)
            boxes = tgt["boxes"].clone()
            boxes[:, [0, 2]] = img.width - tgt["boxes"][:, [2, 0]]
            tgt = {**tgt, "boxes": boxes}
        return img, tgt


def resized_hw(w: int, h: int, short: int, max_size: int) -> tuple[int, int]:
    """Output ``(h, w)`` with the short side at ``short`` and the long side capped at ``max_size``."""
    scale = short / min(w, h)
    if max(w, h) * scale > max_size:
        scale = max_size / max(w, h)
    return int(round(h * scale)), int(round(w * scale))


class RandomResize:
    def __init__(self, sizes: Sequence[int], max_size: int):
        self.sizes, self.max_size = list(sizes), max_size

    def __call__(self, img, tgt):
        new_h, new_w = resized_hw(img.width, img.height, random.choice(self.sizes), self.max_size)
        sx, sy = new_w / img.width, new_h / img.height
        img = img.resize((new_w, new_h), Image.BILINEAR)
        return img, {**tgt, "boxes": tgt["boxes"] * torch.tensor([sx, sy, sx, sy])}


class RandomSizeCrop:
    """Random crop with sides in ``[min_size, max_size]``; retries until at least one box survives."""

    def __init__(self, min_size: int, max_size: int, tries: int = 10):
        self.min_size, self.max_size, self.tries = min_size, max_size, tries

    def __call__(self, img, tgt):
        W, H = img.size
        for _ in range(self.tries):
            w = random.randint(min(self.min_size, W), min(W, self.max_size))
            h = random.randint(min(self.min_size, H), min(H, self.max_size))
            j, i = random.randint(0, W - w), random.randint(0, H - h)
            boxes = tgt["boxes"] - torch.tensor([j, i, j, i], dtype=torch.float32)
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            keep = (boxes[:, 2] > boxes[:, 0] + 1) & (boxes[:, 3] > boxes[:, 1] + 1)
            if keep.any():
                img = img.crop((j, i, j + w, i + h))
                tgt = {**tgt, "boxes": boxes[keep], "labels": tgt["labels"][keep],
                       "difficult": tgt["difficult"][keep]}
                return img, tgt
        return img, tgt


class RandomSelect:
    def __init__(self, t1, t2, p: float = 0.5):
        self.t1, self.t2, self.p = t1, t2, p

    def __call__(self, img, tgt):
        return self.t1(img, tgt) if random.random() < self.p else self.t2(img, tgt)


class PhotometricJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, tgt):
        return self.jitter(img), tgt


class ToTensorNormalize:
    def __call__(self, img, tgt):
        w, h = img.size
        x = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        b = tgt["boxes"]
        cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2 / w, (b[:, 1] + b[:, 3]) / 2 / h,
                              (b[:, 2] - b[:, 0]) / w, (b[:, 3] - b[:, 1]) / h], dim=-1)
        return x, {**tgt, "boxes": cxcywh.reshape(-1, 4), "size": torch.tensor([h, w])}


def build_transforms(train: bool, train_scales: Sequence[int], max_size: int, eval_size: int):
    """DETR-style recipe: flip, then either multi-scale resize or resize -> crop -> resize."""
    if train:
        return Compose([
            RandomHorizontalFlip(),
            RandomSelect(
                RandomResize(train_scales, max_size),
                Compose([RandomResize([300, 350, 400], 1000),
                         RandomSizeCrop(240, 400),
                         RandomResize(train_scales, max_size)]),
            ),
            PhotometricJitter(),
            ToTensorNormalize(),
        ])
    return Compose([RandomResize([eval_size], max_size), ToTensorNormalize()])
