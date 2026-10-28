import numpy as np
import torch

from t1seg.model import UNet
from t1seg.pipeline import run_slice
from t1seg.polar import from_polar, to_polar
from t1seg.postprocess import clean_mask, refine_with_reference, shape_ok
from t1seg.synthetic import ir_signal, make_slice, molli_tis, stone_tis
from t1seg.t1fit import fit_2p, fit_3p


def annulus(cy=128, cx=128, r0=20, r1=28, n=256):
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(yy - cy, xx - cx)
    return ((r >= r0) & (r <= r1)).astype(np.uint8)


def test_unet_shape():
    assert UNet()(torch.zeros(1, 1, 256, 256)).shape == (1, 2, 256, 256)


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
    img = (xx + 2 * yy).astype(float)  # smooth ramp
    p, g = to_polar(img, m)
    back = from_polar(p, m, g)
    assert p.shape == (20, 360) and np.nanmax(np.abs(back - img)[m > 0]) < 3


def test_shape_qc_and_refine():
    good = annulus()
    assert shape_ok(good)
    broken = good.copy()
    broken[:, 128:131] = 0          # cut the ring in two -> Euler != 0
    broken = clean_mask(broken)
    assert not shape_ok(broken)
    shifted = annulus(cy=133, cx=125)
    shifted[:, 128:131] = 0
    ref = refine_with_reference(clean_mask(shifted), good)
    assert shape_ok(ref)


def test_end_to_end_with_gt_masks():
    rng = np.random.default_rng(1)
    imgs, masks, tis, t1 = make_slice(rng)
    res = run_slice(imgs, tis, masks=masks)
    assert res["status"] == "ok"
    m = res["midwall"] & np.isfinite(res["t1_map"])
    t1 = t1[res["ref_index"]]
    assert np.abs(res["t1_map"][m] - t1[m]).mean() < 25
