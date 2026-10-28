"""Synthetic short-axis T1-weighted series for smoke-testing the pipeline.

Not realistic anatomy -- just enough (body, LV blood pool, myocardial annulus,
RV, fat, inter-image motion, Rician noise) to exercise every stage end to end
before real data is available.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# Native T1 at 1.5 T (ms), roughly
T1_BLOOD, T1_MYO, T1_LIVER, T1_FAT, T1_MUSCLE = 1550.0, 1000.0, 600.0, 260.0, 1000.0


def stone_tis(rr=1000.0):
    """The paper's STONE TI scheme: inf, 115 + k*RR, 350 + k*RR (k = 0..4)."""
    return np.array([np.inf] + [115 + k * rr for k in range(5)] + [350 + k * rr for k in range(5)])


def molli_tis(rr=1000.0):
    """MOLLI 5(3)3 TI scheme (approx.)."""
    return np.sort(np.array([100 + k * rr for k in range(5)] + [180 + k * rr for k in range(3)]))


def ir_signal(t1, ti, m0=1.0):
    """Ideal 2-parameter inversion recovery magnitude."""
    return np.abs(m0 * (1 - 2 * np.exp(-ti / t1)))


def _ellipse(yy, xx, cy, cx, ry, rx, ang=0.0):
    c, s = np.cos(ang), np.sin(ang)
    y, x = yy - cy, xx - cx
    return ((c * x + s * y) / rx) ** 2 + ((-s * x + c * y) / ry) ** 2 <= 1


def make_slice(rng, tis=None, size=256, motion_px=2.0, noise=0.02):
    """Returns images (K,H,W), masks (K,H,W), tis (K,), t1_true (K,H,W) (moves with each frame)."""
    tis = stone_tis() if tis is None else np.asarray(tis, float)
    yy, xx = np.mgrid[:size, :size].astype(float)

    cy, cx = size / 2 + rng.uniform(-25, 25, 2)
    r_endo = rng.uniform(12, 24)
    thick = rng.uniform(5, 12)
    ecc = rng.uniform(0.85, 1.0)
    ang = rng.uniform(0, np.pi)

    def anatomy(dy, dx):
        body = _ellipse(yy, xx, size / 2, size / 2, size * 0.36, size * 0.46)
        fat = body & ~_ellipse(yy, xx, size / 2, size / 2, size * 0.33, size * 0.43)
        liver = _ellipse(yy, xx, size * 0.75, size * 0.3, 40, 55) & body
        epi = _ellipse(yy, xx, cy + dy, cx + dx, (r_endo + thick) * ecc, r_endo + thick, ang)
        endo = _ellipse(yy, xx, cy + dy, cx + dx, r_endo * ecc, r_endo, ang)
        rv = _ellipse(yy, xx, cy + dy - 5, cx + dx - r_endo - thick - 14, r_endo * 1.2, 14, ang) & ~epi
        myo = epi & ~endo
        t1 = np.full((size, size), T1_MUSCLE)
        t1[liver] = T1_LIVER
        t1[fat] = T1_FAT
        t1[rv | endo] = T1_BLOOD
        t1[myo] = T1_MYO
        m0 = body.astype(float) * 1.0
        m0[fat] = 1.3
        return t1, m0, myo

    # smooth regional T1 variation in the myocardium (e.g. a "diffuse fibrosis" patch)
    patch = gaussian_filter(rng.standard_normal((size, size)), 12)
    patch = 120 * patch / (np.abs(patch).max() + 1e-9)

    images, masks, t1s = [], [], []
    for k, ti in enumerate(tis):
        d = rng.normal(0, motion_px, 2) if k > 0 else (0.0, 0.0)
        t1, m0, myo = anatomy(*d)
        t1 = t1 + patch * myo
        t1s.append(t1.astype(np.float32))
        s = m0 * ir_signal(t1, ti)
        s = gaussian_filter(s, 0.7)
        s = np.abs(s + noise * rng.standard_normal(s.shape) + 1j * noise * rng.standard_normal(s.shape))
        images.append(s.astype(np.float32))
        masks.append(myo.astype(np.uint8))
    return np.stack(images), np.stack(masks), tis, np.stack(t1s)
