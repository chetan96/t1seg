"""Segmentation losses. Configs give a list of weighted terms:

    loss:
      - {name: ce, weight: 1.0}
      - {name: dice, weight: 0.5}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import LOSSES


@LOSSES.register("ce")
class CrossEntropy(nn.Module):
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        self.ls = label_smoothing

    def forward(self, logits, target):
        w = None if self.w is None else self.w.to(logits.device)
        return F.cross_entropy(logits, target, weight=w, label_smoothing=self.ls)


@LOSSES.register("dice")
class SoftDice(nn.Module):
    """Soft Dice over foreground classes (1..C-1)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = logits.softmax(1)[:, 1:]
        t = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].float()
        inter = (p * t).sum((2, 3))
        return (1 - (2 * inter + self.smooth) / (p.sum((2, 3)) + t.sum((2, 3)) + self.smooth)).mean()


@LOSSES.register("focal")
class Focal(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, 1).gather(1, target[:, None])[:, 0]
        return (-(1 - logp.exp()) ** self.gamma * logp).mean()


class WeightedSum(nn.Module):
    def __init__(self, terms):
        super().__init__()
        self.terms = nn.ModuleList(t for t, _ in terms)
        self.weights = [w for _, w in terms]

    def forward(self, logits, target):
        return sum(w * t(logits, target) for t, w in zip(self.terms, self.weights))


def build_loss(cfg):
    """cfg: a name, a dict, or a list of dicts with optional 'weight'."""
    if not isinstance(cfg, list):
        cfg = [cfg]
    terms = []
    for c in cfg:
        c = {"name": c} if isinstance(c, str) else dict(c)
        w = c.pop("weight", 1.0)
        terms.append((LOSSES.build(c), w))
    return WeightedSum(terms)
