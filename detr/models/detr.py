"""The DETR detector."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from detr.config import ModelConfig
from detr.models.backbone import Backbone
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.transformer import Transformer


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x) if i == len(self.layers) - 1 else F.relu(layer(x))
        return x


class DETR(nn.Module):
    """Backbone -> 1x1 projection -> transformer -> class / box heads on every decoder layer.

    ``forward`` returns a dict with ``pred_logits [B, Q, C+1]`` and ``pred_boxes [B, Q, 4]``
    (normalised cxcywh) from the last decoder layer, plus ``aux_outputs`` for the others.
    """

    def __init__(self, num_classes: int, num_queries: int = 20, d_model: int = 256, n_heads: int = 8,
                 enc_layers: int = 4, dec_layers: int = 4, dim_ffn: int = 1024, dropout: float = 0.1,
                 backbone: str = "resnet50", pretrained_backbone: bool = True,
                 train_backbone_layers=("layer2", "layer3", "layer4")):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Backbone(backbone, pretrained_backbone, train_backbone_layers)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        self.transformer = Transformer(d_model, n_heads, enc_layers, dec_layers, dim_ffn, dropout)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1: "no object"
        self.box_head = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, images: Tensor, mask: Tensor, return_attn: bool = False) -> dict:
        feat, feat_mask = self.backbone(images, mask)
        pos = self.pos_embed(feat_mask)
        src = self.input_proj(feat)
        B, _, H, W = src.shape

        src = src.flatten(2).transpose(1, 2)
        pos = pos.flatten(2).transpose(1, 2).to(src.dtype)
        query_pos = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        hs, attn = self.transformer(src, feat_mask.flatten(1), pos, query_pos, return_attn)
        logits = self.class_head(hs)
        boxes = self.box_head(hs).sigmoid()

        out = {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "aux_outputs": [{"pred_logits": lg, "pred_boxes": bx} for lg, bx in zip(logits[:-1], boxes[:-1])],
        }
        if return_attn:
            out.update(attn=attn, feat_hw=(H, W), feat_mask=feat_mask)
        return out

    def param_groups(self, lr: float, lr_backbone: float) -> list[dict]:
        backbone, rest = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (backbone if name.startswith("backbone.") else rest).append(p)
        return [{"params": rest, "lr": lr}, {"params": backbone, "lr": lr_backbone}]


def build_model(cfg: ModelConfig, num_classes: int) -> DETR:
    return DETR(
        num_classes=num_classes,
        num_queries=cfg.num_queries,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        dim_ffn=cfg.dim_ffn,
        dropout=cfg.dropout,
        backbone=cfg.backbone,
        pretrained_backbone=cfg.pretrained_backbone,
        train_backbone_layers=cfg.train_backbone_layers,
    )
