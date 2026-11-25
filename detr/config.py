"""Typed experiment configuration, loaded from YAML with optional command-line overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "~/datasets/VOC"
    classes: list[str] = field(default_factory=lambda: ["bird", "cat", "dog"])
    val_size: int = 125
    train_scales: list[int] = field(default_factory=lambda: [288, 320, 352, 384, 416, 448, 480])
    max_size: int = 640
    eval_size: int = 416
    num_workers: int = 4


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained_backbone: bool = True
    train_backbone_layers: list[str] = field(default_factory=lambda: ["layer2", "layer3", "layer4"])
    d_model: int = 256
    n_heads: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    dim_ffn: int = 1024
    dropout: float = 0.1
    num_queries: int = 20


@dataclass
class LossConfig:
    cost_class: float = 1.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    eos_coef: float = 0.1
    aux_loss: bool = True


@dataclass
class TrainConfig:
    output_dir: str = "runs/voc_bird_cat_dog"
    epochs: int = 100
    lr_drop_epoch: int = 75
    batch_size: int = 8
    accum_steps: int = 1
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    clip_norm: float = 0.1
    warmup_iters: int = 300
    amp: bool = True
    eval_every: int = 5
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        return _from_dict(cls, d)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> Config:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for item in overrides or []:
            _apply_override(raw, item)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def data_root(self) -> Path:
        return Path(self.data.root).expanduser()


def _from_dict(cls, d: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in d.items():
        f_type = known[name].default_factory if callable(known[name].default_factory) else None
        default = f_type() if f_type else None
        kwargs[name] = _from_dict(type(default), value) if is_dataclass(default) else value
    return cls(**kwargs)


def _apply_override(raw: dict[str, Any], item: str) -> None:
    """Apply a ``section.key=value`` override; the value is parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"Override must look like section.key=value, got {item!r}")
    key, value = item.split("=", 1)
    node = raw
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)
