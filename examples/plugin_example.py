"""Template: adding new methods WITHOUT touching t1seg's code.

Enable with `plugins: [examples/plugin_example.py]` in a YAML (or `--set plugins=[examples/plugin_example.py]`),
then refer to the new names, e.g.
    model: {name: tiny_fcn, width: 16}
    pipeline.postprocess: [{name: largest_component}, {name: binary_closing, radius: 2}, {name: shape_qc}]
    pipeline.fitter: {name: 2p_scipy}

Each registry documents its interface in t1seg/registry.py. A checkpoint trained with a plugin model
records the plugin list, so loading it later re-imports the plugin automatically.
"""
import numpy as np
import torch.nn as nn
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

from t1seg.registry import FITTERS, MODELS, POSTPROCESS
from t1seg.t1fit import fit_map


# 1) a new segmentation architecture: (B, in_channels, H, W) -> logits (B, n_classes, H, W)
@MODELS.register("tiny_fcn")
class TinyFCN(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(width, n_classes, 1))

    def forward(self, x):
        return self.net(x)


# 2) a new post-processing step: factory(**params) -> f(masks, ok, tis, pixel_spacing) -> (masks, ok)
@POSTPROCESS.register("binary_closing")
def binary_closing(radius=1):
    st = ndi.generate_binary_structure(2, 1)

    def f(masks, ok, tis, pixel_spacing):
        return np.stack([ndi.binary_closing(m, st, iterations=radius) for m in masks]).astype(np.uint8), ok
    return f


# 3) a new T1 estimator: callable(stack (K, ...), tis) -> T1 (...)
@FITTERS.register("2p_scipy")
class ScipyTwoParam:
    """Reference implementation with scipy.curve_fit (slow, for cross-checking the vectorised fitter)."""

    def __call__(self, stack, tis):
        model = lambda ti, a, t1: np.abs(a * (1 - 2 * np.exp(-ti / t1)))  # noqa: E731

        def fit(sig, tis):
            out = np.empty(sig.shape[1])
            for i in range(sig.shape[1]):
                try:
                    out[i] = curve_fit(model, tis, sig[:, i], p0=(sig[:, i].max(), 1000.0), maxfev=2000)[0][1]
                except RuntimeError:
                    out[i] = np.nan
            return out
        return fit_map(stack, tis, fit)
