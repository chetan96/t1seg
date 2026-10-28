"""Component registries -- the plug-and-play backbone of t1seg.

Every swappable stage (model, loss, dataset, segmenter, post-processing step,
aligner, T1 fitter) lives in a registry. Configs refer to components by name:

    model: {name: unet, base: 32}          ->  MODELS.build(cfg)  ==  UNet(base=32)
    fitter: 3p                             ->  FITTERS.build("3p")

To add a new method (e.g. from a paper you just read), write it anywhere and
decorate it -- no existing code changes:

    from t1seg.registry import MODELS

    @MODELS.register("my_transunet")
    class MyTransUNet(nn.Module): ...

then list the module under `plugins:` in the YAML so it is imported.
"""
import importlib
import importlib.util
import re
import sys
from pathlib import Path


class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}

    def register(self, name=None, obj=None):
        """Use as ``@REG.register("name")`` or ``REG.register("name", obj)``."""
        def deco(o):
            key = name or o.__name__
            if key in self._items and self._items[key] is not o:
                raise KeyError(f"{self.kind} '{key}' is already registered ({self._items[key]})")
            self._items[key] = o
            return o
        return deco(obj) if obj is not None else deco

    def get(self, name):
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def build(self, cfg, **extra):
        """cfg is a name, or a dict with 'name' plus constructor kwargs."""
        if isinstance(cfg, str):
            cfg = {"name": cfg}
        cfg = dict(cfg)
        return self.get(cfg.pop("name"))(**cfg, **extra)

    def names(self):
        return sorted(self._items)

    def __contains__(self, name):
        return name in self._items

    def __repr__(self):
        return f"Registry({self.kind}: {self.names()})"


MODELS = Registry("model")             # nn.Module: (B, C_in, H, W) -> logits (B, n_classes, H, W)
LOSSES = Registry("loss")              # f(logits, target) -> scalar
DATASETS = Registry("dataset")         # torch Dataset yielding (image (C,H,W), mask (H,W))
SEGMENTERS = Registry("segmenter")     # f(images (K,H,W), tis, **ctx) -> masks (K,H,W)
POSTPROCESS = Registry("postprocess")  # f(masks, ok, tis, pixel_spacing) -> (masks, ok)
ALIGNERS = Registry("aligner")         # .align(images, masks, ok) -> Aligned
FITTERS = Registry("fitter")           # f(stack (K, ...), tis) -> T1 (...)

ALL = dict(model=MODELS, loss=LOSSES, dataset=DATASETS, segmenter=SEGMENTERS,
           postprocess=POSTPROCESS, aligner=ALIGNERS, fitter=FITTERS)


def import_plugins(modules):
    """Import user modules (dotted names or .py paths) so their @register calls run."""
    for m in modules or []:
        if m.endswith(".py"):
            path = Path(m).resolve()
            name = "t1seg_plugin_" + re.sub(r"\W", "_", str(path.with_suffix("")))
            if name in sys.modules:  # idempotent: a plugin file is executed only once
                continue
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                del sys.modules[name]
                raise
        else:
            importlib.import_module(m)


def describe():
    return "\n".join(f"{k:12s} {r.names()}" for k, r in ALL.items())
