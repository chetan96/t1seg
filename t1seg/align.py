"""Aligners: put the myocardium of every T1w image on a common grid before fitting.

Interface:  aligner.align(images (K,H,W), masks (K,H,W), ok (K,), tis (K,)) -> Aligned
  Aligned.stack     (K', ...)   samples per used image, on the common grid
  Aligned.tis       (K',)
  Aligned.to_image  f(map on common grid) -> Cartesian (H,W) map on the reference image
  Aligned.ref_index, Aligned.ref_mask

Add e.g. a deformable-registration or deep-learning motion-correction aligner by
registering a class with the same interface.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .polar import from_polar, to_polar
from .registry import ALIGNERS


@dataclass
class Aligned:
    stack: np.ndarray
    tis: np.ndarray
    to_image: Callable
    ref_index: int
    ref_mask: np.ndarray


def _pick_reference(idx, tis, reference):
    t = tis[idx]
    if reference == "shortest_ti":
        return int(idx[np.argmin(t)])
    if reference == "longest_ti":
        return int(idx[np.argmax(t)])
    return int(reference)  # explicit image index


@ALIGNERS.register("polar")
class PolarAligner:
    """Paper: per-image polar resampling about each mask's centre of mass (360 rays x C depths)."""

    def __init__(self, n_angles=360, depth=20, reference="shortest_ti"):
        self.n_angles, self.depth, self.reference = n_angles, depth, reference

    def align(self, images, masks, ok, tis):
        idx = np.where(ok)[0]
        stack = np.stack([to_polar(images[k], masks[k], self.n_angles, self.depth)[0] for k in idx])
        k_ref = _pick_reference(idx, tis, self.reference)
        _, geom = to_polar(masks[k_ref].astype(float), masks[k_ref], self.n_angles, self.depth)
        return Aligned(stack, tis[idx], lambda m: from_polar(m, masks[k_ref], geom), k_ref, masks[k_ref])


@ALIGNERS.register("none")
class NoAlignment:
    """Fit directly on the reference mask's pixels, assuming no motion (e.g. already motion-corrected
    maps from the scanner). Useful baseline to quantify what alignment buys you."""

    def __init__(self, reference="shortest_ti"):
        self.reference = reference

    def align(self, images, masks, ok, tis):
        idx = np.where(ok)[0]
        k_ref = _pick_reference(idx, tis, self.reference)
        ref = masks[k_ref] > 0
        stack = images[idx][:, ref]

        def to_image(v):
            out = np.full(ref.shape, np.nan)
            out[ref] = v
            return out

        return Aligned(stack, tis[idx], to_image, k_ref, masks[k_ref])
