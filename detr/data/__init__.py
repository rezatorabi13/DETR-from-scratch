"""Data loading: VOC subsets, box-aware transforms and padded batching."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from detr.config import Config
from detr.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from detr.data.voc import VOCDetection, build_splits, download_voc2007

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "build_datasets", "build_loader", "collate_fn"]


def collate_fn(batch, stride: int = 32):
    """Pad images to a common size (multiple of ``stride``) and return a padding mask.

    Returns ``(images [B, 3, H, W], mask [B, H, W] with True = padding, list of targets)``.
    """
    images, targets = zip(*batch)
    H = math.ceil(max(img.shape[1] for img in images) / stride) * stride
    W = math.ceil(max(img.shape[2] for img in images) / stride) * stride
    padded = images[0].new_zeros(len(images), 3, H, W)
    mask = torch.ones(len(images), H, W, dtype=torch.bool)
    for k, img in enumerate(images):
        padded[k, :, : img.shape[1], : img.shape[2]] = img
        mask[k, : img.shape[1], : img.shape[2]] = False
    return padded, mask, list(targets)


def build_datasets(cfg: Config):
    voc_dir = download_voc2007(cfg.data_root)
    train, val, test = build_splits(voc_dir, cfg.data.classes, cfg.data.val_size, cfg.train.seed)
    tf_args = (cfg.data.train_scales, cfg.data.max_size, cfg.data.eval_size)
    return (
        VOCDetection(train, voc_dir, cfg.data.classes, build_transforms(True, *tf_args)),
        VOCDetection(val, voc_dir, cfg.data.classes, build_transforms(False, *tf_args)),
        VOCDetection(test, voc_dir, cfg.data.classes, build_transforms(False, *tf_args)),
    )


def build_loader(dataset, batch_size: int, train: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
