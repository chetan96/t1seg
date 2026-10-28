"""YAML configs with command-line overrides and plugin loading."""
import copy

import yaml

from .registry import import_plugins


def _parse_value(v):
    return yaml.safe_load(v)  # "30" -> 30, "true" -> True, "[1,2]" -> list, "null" -> None


def apply_overrides(cfg, overrides):
    """overrides: ["train.epochs=30", "model.name=smp", "fitter=3p"], applied left to right.

    Setting `<block>.name` to a *different* component resets that block to {name: ...}, since the old
    component's kwargs rarely apply to the new one; add its kwargs with further overrides, e.g.
    `model.name=smp model.encoder_name=resnet50`."""
    cfg = copy.deepcopy(cfg)
    for o in overrides or []:
        key, _, val = o.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            if not isinstance(node.get(p), dict):
                node[p] = {} if node.get(p) is None else {"name": node[p]}
            node = node[p]
        val = _parse_value(val)
        if leaf == "name" and node.get("name") not in (None, val):
            node.clear()
        node[leaf] = val
    return cfg


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg = apply_overrides(cfg, overrides)
    import_plugins(cfg.get("plugins"))
    return cfg
