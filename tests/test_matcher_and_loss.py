import torch

from detr.losses import SetCriterion
from detr.matcher import HungarianMatcher


def _perfect_outputs(targets, num_queries=6, num_classes=3):
    """Outputs whose queries 2, 4, ... exactly reproduce each target box with a confident class."""
    B = len(targets)
    logits = torch.full((B, num_queries, num_classes + 1), -10.0)
    logits[..., -1] = 10.0
    boxes = torch.full((B, num_queries, 4), 0.5)
    for b, t in enumerate(targets):
        for k, (box, lbl) in enumerate(zip(t["boxes"], t["labels"])):
            q = 2 * k + 1
            boxes[b, q] = box
            logits[b, q, :] = -10.0
            logits[b, q, lbl] = 10.0
    return {"pred_logits": logits, "pred_boxes": boxes, "aux_outputs": []}


TARGETS = [
    {"boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.3, 0.4]]), "labels": torch.tensor([0, 2])},
    {"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]), "labels": torch.tensor([1])},
]


def test_matcher_recovers_assignment():
    indices = HungarianMatcher()(_perfect_outputs(TARGETS), TARGETS)
    for (src, tgt), t in zip(indices, TARGETS):
        assert sorted(zip(tgt.tolist(), src.tolist())) == [(k, 2 * k + 1) for k in range(len(t["labels"]))]


def test_matcher_handles_empty_targets():
    targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    out = {"pred_logits": torch.randn(1, 5, 4), "pred_boxes": torch.rand(1, 5, 4)}
    src, tgt = HungarianMatcher()(out, targets)[0]
    assert len(src) == len(tgt) == 0


def test_loss_is_near_zero_for_perfect_predictions():
    criterion = SetCriterion(3, HungarianMatcher(), eos_coef=0.1, aux_loss=False)
    losses = criterion(_perfect_outputs(TARGETS), TARGETS)
    assert losses["loss"] < 1e-3
    assert losses["match_acc"] == 1.0


def test_aux_loss_adds_one_term_per_layer():
    out = _perfect_outputs(TARGETS)
    out["aux_outputs"] = [{"pred_logits": torch.randn_like(out["pred_logits"]),
                           "pred_boxes": torch.rand_like(out["pred_boxes"])} for _ in range(3)]
    losses = SetCriterion(3, HungarianMatcher(), aux_loss=True)(out, TARGETS)
    assert len(losses["layer_losses"]) == 4
