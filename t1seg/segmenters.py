"""Segmenters: anything that turns a slice's T1w images into myocardium masks.

Interface:  segmenter(images (K,H,W), tis (K,), **ctx) -> masks (K,H,W) uint8
`ctx` may hold e.g. gt `masks` or `pixel_spacing`. Wrap any method this way --
a trained PyTorch net, nnU-Net / MONAI bundle predictor, a classical method, or
ground truth for testing the downstream stages in isolation.
"""
import numpy as np
import torch

from .data import normalize
from .device import resolve_device
from .io import pad_crop
from .registry import SEGMENTERS


@SEGMENTERS.register("torch")
class TorchSegmenter:
    """A checkpoint from scripts/train.py (any registered architecture)."""

    def __init__(self, checkpoint, device="auto", size=256, normalization="zscore", batch_size=16):
        from .models import load_checkpoint

        self.device = resolve_device(device)
        self.model, ck = load_checkpoint(checkpoint, self.device)
        self.size, self.normalization, self.bs = size, normalization, batch_size

    @torch.no_grad()
    def __call__(self, images, tis=None, **ctx):
        x = np.stack([normalize(im, self.normalization) for im in pad_crop(images, self.size)])
        x = torch.from_numpy(x)[:, None]
        out = [self.model(x[i:i + self.bs].to(self.device)).argmax(1).cpu() for i in range(0, len(x), self.bs)]
        return torch.cat(out).numpy().astype(np.uint8)


@SEGMENTERS.register("ground_truth")
class GroundTruthSegmenter:
    """Returns the masks stored with the data -- isolates alignment/fitting from segmentation errors."""

    def __init__(self, size=256):
        self.size = size

    def __call__(self, images, tis=None, masks=None, **ctx):
        if masks is None:
            raise ValueError("ground_truth segmenter needs `masks` in the input data")
        return pad_crop(np.asarray(masks, np.uint8), self.size)
