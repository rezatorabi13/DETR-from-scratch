"""Post-norm encoder-decoder transformer with padding masks and intermediate decoder outputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _ffn(d: int, dim_ffn: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d, dim_ffn), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ffn, d))


class EncoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dim_ffn: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = _ffn(d, dim_ffn, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src: Tensor, pos: Tensor, pad_mask: Tensor) -> Tensor:
        q = k = src + pos
        attn = self.self_attn(q, k, src, key_padding_mask=pad_mask, need_weights=False)[0]
        src = self.norm1(src + self.drop1(attn))
        return self.norm2(src + self.drop2(self.ffn(src)))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dim_ffn: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = _ffn(d, dim_ffn, dropout)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop1, self.drop2, self.drop3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, mem_pad_mask: Tensor, mem_pos: Tensor,
                query_pos: Tensor, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        q = k = tgt + query_pos
        tgt = self.norm1(tgt + self.drop1(self.self_attn(q, k, tgt, need_weights=False)[0]))
        out, attn = self.cross_attn(tgt + query_pos, memory + mem_pos, memory,
                                    key_padding_mask=mem_pad_mask,
                                    need_weights=return_attn, average_attn_weights=False)
        tgt = self.norm2(tgt + self.drop2(out))
        tgt = self.norm3(tgt + self.drop3(self.ffn(tgt)))
        return tgt, attn


class Transformer(nn.Module):
    def __init__(self, d: int = 256, n_heads: int = 8, enc_layers: int = 4, dec_layers: int = 4,
                 dim_ffn: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.ModuleList(EncoderLayer(d, n_heads, dim_ffn, dropout) for _ in range(enc_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d, n_heads, dim_ffn, dropout) for _ in range(dec_layers))
        self.dec_norm = nn.LayerNorm(d)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, pad_mask: Tensor, pos: Tensor, query_pos: Tensor,
                return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        """Returns decoder outputs of every layer ``[L, B, Q, d]`` and, optionally, the last
        layer's per-head cross-attention ``[B, heads, Q, H*W]``."""
        memory = src
        for layer in self.encoder:
            memory = layer(memory, pos, pad_mask)
        tgt = torch.zeros_like(query_pos)
        hs, attn = [], None
        for layer in self.decoder:
            tgt, attn = layer(tgt, memory, pad_mask, pos, query_pos, return_attn)
            hs.append(self.dec_norm(tgt))
        return torch.stack(hs), attn
