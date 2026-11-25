import pytest
import torch

from detr.config import Config
from detr.losses import SetCriterion
from detr.matcher import HungarianMatcher
from detr.models import DETR
from detr.models.position_encoding import PositionEmbeddingSine
from detr.models.transformer import Transformer


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return DETR(num_classes=3, num_queries=5, d_model=64, n_heads=4, enc_layers=1, dec_layers=2,
                dim_ffn=128, backbone="resnet18", pretrained_backbone=False)


def test_forward_shapes(tiny_model):
    images = torch.randn(2, 3, 128, 160)
    mask = torch.zeros(2, 128, 160, dtype=torch.bool)
    out = tiny_model(images, mask, return_attn=True)
    assert out["pred_logits"].shape == (2, 5, 4)
    assert out["pred_boxes"].shape == (2, 5, 4)
    assert ((out["pred_boxes"] >= 0) & (out["pred_boxes"] <= 1)).all()
    assert len(out["aux_outputs"]) == 1
    assert out["feat_hw"] == (4, 5)
    assert out["attn"].shape == (2, 4, 5, 20)


def test_transformer_ignores_padded_tokens():
    torch.manual_seed(0)
    tf = Transformer(d=32, n_heads=4, enc_layers=2, dec_layers=2, dim_ffn=64, dropout=0.0).eval()
    src, pos, queries = torch.randn(1, 10, 32), torch.randn(1, 10, 32), torch.randn(1, 3, 32)
    pad = torch.zeros(1, 10, dtype=torch.bool)
    pad[:, 7:] = True
    noisy = src.clone()
    noisy[:, 7:] = torch.randn(1, 3, 32) * 100  # garbage in the padded positions
    with torch.no_grad():
        a, _ = tf(src, pad, pos, queries)
        b, _ = tf(noisy, pad, pos, queries)
    assert torch.allclose(a, b, atol=1e-5)


def test_position_encoding_ignores_padding():
    pe = PositionEmbeddingSine(32)
    small = torch.zeros(1, 6, 8, dtype=torch.bool)
    big = torch.ones(1, 10, 12, dtype=torch.bool)
    big[:, :6, :8] = False
    assert torch.allclose(pe(small), pe(big)[..., :6, :8])


def test_backward_pass_reaches_trainable_parameters(tiny_model):
    tiny_model.train()
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]), "labels": torch.tensor([1])}]
    criterion = SetCriterion(3, HungarianMatcher())
    out = tiny_model(torch.randn(1, 3, 96, 96), torch.zeros(1, 96, 96, dtype=torch.bool))
    criterion(out, targets)["loss"].backward()
    assert tiny_model.query_embed.weight.grad is not None
    assert tiny_model.backbone.body[0].weight.grad is None  # stem is frozen


def test_config_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("train:\n  epochs: 10\nmodel:\n  num_queries: 7\n")
    cfg = Config.from_yaml(path, ["train.epochs=3", "data.classes=[cat, dog]"])
    assert cfg.train.epochs == 3 and cfg.model.num_queries == 7 and cfg.data.classes == ["cat", "dog"]
    with pytest.raises(KeyError):
        Config.from_dict({"train": {"not_a_key": 1}})
