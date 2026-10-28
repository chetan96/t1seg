import numpy as np
import pytest
import torch

import t1seg  # noqa: F401  registers built-ins
from t1seg.config import apply_overrides
from t1seg.device import resolve_device
from t1seg.losses import build_loss
from t1seg.models import build_model, load_checkpoint, save_checkpoint
from t1seg.pipeline import Pipeline
from t1seg.polar import from_polar, to_polar
from t1seg.postprocess import clean_mask, refine_with_reference, shape_ok
from t1seg.registry import FITTERS, MODELS, POSTPROCESS, Registry, import_plugins
from t1seg.synthetic import ir_signal, make_slice, molli_tis, stone_tis
from t1seg.t1fit import fit_2p, fit_3p


def annulus(cy=128, cx=128, r0=20, r1=28, n=256):
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - cy, xx - cx)
    return ((r >= r0) & (r <= r1)).astype(np.uint8)


# ---- framework ------------------------------------------------------------------------------

def test_registry_build_and_errors():
    r = Registry("thing")
    r.register("double", lambda x=1: 2 * x)
    assert r.build({"name": "double", "x": 3}) == 6 and r.build("double") == 2
    with pytest.raises(KeyError, match="Available"):
        r.get("nope")


def test_overrides():
    cfg = {"train": {"epochs": 3}, "fitter": "2p"}
    out = apply_overrides(cfg, ["train.epochs=30", "fitter.name=3p", "device=cpu", "pipeline.postprocess=[]"])
    assert out["train"]["epochs"] == 30 and out["fitter"] == {"name": "3p"} and out["device"] == "cpu"
    assert out["pipeline"]["postprocess"] == [] and cfg["train"]["epochs"] == 3
    seg = {"segmenter": {"name": "torch", "checkpoint": "x.pt"}}
    assert apply_overrides(seg, ["segmenter.name=ground_truth"])["segmenter"] == {"name": "ground_truth"}
    assert apply_overrides(seg, ["segmenter.name=torch"])["segmenter"]["checkpoint"] == "x.pt"


def test_device_auto_and_explicit():
    assert resolve_device("auto").type in ("cuda", "mps", "cpu")
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="not available"):
            resolve_device("cuda")


@pytest.mark.parametrize("cfg", [{"name": "unet", "base": 8, "depth": 2},
                                 {"name": "torchvision_deeplab", "backbone": "resnet50"}])
def test_models_shape(cfg):
    m = build_model(cfg).eval()
    assert m(torch.zeros(1, 1, 64, 64)).shape == (1, 2, 64, 64)


def test_loss_combo():
    loss = build_loss([{"name": "ce"}, {"name": "dice", "weight": 0.5}, "focal"])
    logits, y = torch.randn(2, 2, 8, 8, requires_grad=True), torch.randint(0, 2, (2, 8, 8))
    loss(logits, y).backward()


def test_checkpoint_roundtrip_with_plugin(tmp_path):
    import_plugins(["examples/plugin_example.py"])
    assert "tiny_fcn" in MODELS and "binary_closing" in POSTPROCESS and "2p_scipy" in FITTERS
    cfg = {"name": "tiny_fcn", "width": 4}
    m = build_model(cfg)
    save_checkpoint(tmp_path / "m.pt", m, cfg, plugins=["examples/plugin_example.py"])
    m2, _ = load_checkpoint(tmp_path / "m.pt")
    x = torch.randn(1, 1, 32, 32)
    assert torch.allclose(m.eval()(x), m2(x))


# ---- methods --------------------------------------------------------------------------------

def test_fit_2p_recovers_t1():
    tis = stone_tis()
    true = np.array([600.0, 1000.0, 1500.0])
    sig = np.stack([ir_signal(true, ti) for ti in tis]) * 3.0
    t1, a, _ = fit_2p(sig, tis)
    assert np.allclose(t1, true, rtol=0.005) and np.allclose(a, 3.0, rtol=0.01)


def test_fit_3p_recovers_t1():
    tis = molli_tis()
    t1_true, t1s = np.array([1000.0, 1400.0]), np.array([700.0, 900.0])
    a = np.array([1.0, 2.0])
    b = a * (t1_true / t1s + 1)
    sig = np.abs(a - b * np.exp(-tis[:, None] / t1s))
    assert np.allclose(fit_3p(sig, tis)[0], t1_true, rtol=0.01)


def test_polar_roundtrip():
    m = annulus()
    yy, xx = np.mgrid[:256, :256]
    img = (xx + 2 * yy).astype(float)
    p, g = to_polar(img, m)
    back = from_polar(p, m, g)
    assert p.shape == (20, 360) and np.nanmax(np.abs(back - img)[m > 0]) < 3


def test_shape_qc_and_refine():
    good = annulus()
    assert shape_ok(good)
    broken = good.copy()
    broken[:, 128:131] = 0
    assert not shape_ok(clean_mask(broken))
    shifted = annulus(cy=133, cx=125)
    shifted[:, 128:131] = 0
    assert shape_ok(refine_with_reference(clean_mask(shifted), good))


@pytest.mark.parametrize("aligner,tol", [("polar", 25), ("none", 80)])
def test_end_to_end_gt_masks(aligner, tol):
    rng = np.random.default_rng(1)
    imgs, masks, tis, t1 = make_slice(rng)
    pipe = Pipeline.from_config({"segmenter": "ground_truth", "aligner": aligner})
    res = pipe(imgs, tis, masks=masks)
    assert res["status"] == "ok"
    m = res["midwall"] & np.isfinite(res["t1_map"])
    t1 = t1[res["ref_index"]]
    assert np.abs(res["t1_map"][m] - t1[m]).mean() < tol
