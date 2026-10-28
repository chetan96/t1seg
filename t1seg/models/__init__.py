"""Model building + self-describing checkpoints.

A checkpoint stores the model *config* (registry name + kwargs), so any registered
architecture -- built-in, third-party, or your own plugin -- reloads with load_checkpoint().
"""
import torch

from ..registry import MODELS, import_plugins
from . import external, unet  # noqa: F401  (registers built-ins)
from .unet import UNet  # noqa: F401


def build_model(cfg):
    return MODELS.build(cfg)


def save_checkpoint(path, model, model_cfg, plugins=None, **meta):
    torch.save(dict(model=model.state_dict(), model_cfg=dict(model_cfg), plugins=list(plugins or []), **meta),
               path)


def load_checkpoint(path, device="cpu", strict=True):
    """Returns (model in eval mode on `device`, checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    import_plugins(ck.get("plugins"))
    cfg = ck.get("model_cfg") or {"name": "unet", **ck.get("model_kwargs", {})}  # v0.1 checkpoints
    model = build_model(cfg)
    model.load_state_dict(ck["model"], strict=strict)
    return model.to(device).eval(), ck


def init_weights_from(model, path, strict=False):
    """Transfer learning: copy matching tensors (by name and shape) from another checkpoint."""
    src = torch.load(path, map_location="cpu", weights_only=False)
    src = src.get("model", src)
    own = model.state_dict()
    keep = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    if strict and len(keep) != len(own):
        raise RuntimeError(f"only {len(keep)}/{len(own)} tensors matched")
    return len(keep), len(own)
