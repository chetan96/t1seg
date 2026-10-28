"""Post-processing, shape QC and refinement of FCN masks (paper: Methods, 'Post processing')."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage.measure import euler_number, label, regionprops

from .registry import POSTPROCESS

MIN_AREA_CM2 = 5.0
MAX_ECCENTRICITY = 0.65  # max seen in the paper's training set; re-derive on your data


def clean_mask(mask, pixel_spacing=(1.4, 1.4), min_area_cm2=MIN_AREA_CM2):
    """Keep only the largest connected object, and drop it if smaller than min_area."""
    lab = label(mask > 0, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(mask, np.uint8)
    areas = np.bincount(lab.ravel())[1:]
    keep = 1 + int(np.argmax(areas))
    px_cm2 = pixel_spacing[0] * pixel_spacing[1] / 100.0
    if areas[keep - 1] * px_cm2 < min_area_cm2:
        return np.zeros_like(mask, np.uint8)
    return (lab == keep).astype(np.uint8)


def shape_ok(mask, max_ecc=MAX_ECCENTRICITY):
    """Annulus check: Euler number 0 (one object, one hole) and low eccentricity."""
    if mask.sum() == 0:
        return False
    if euler_number(mask > 0, connectivity=2) != 0:
        return False
    return regionprops(mask.astype(np.uint8))[0].eccentricity <= max_ecc


def _warp(mask, params, center):
    ty, tx, rot, logs = params
    c, s = np.cos(rot), np.sin(rot)
    m = np.array([[c, -s], [s, c]]) / np.exp(logs)  # output -> input mapping
    offset = center - m @ (center + np.array([ty, tx]))
    return ndi.affine_transform(mask.astype(float), m, offset=offset, order=1)


def dice(a, b):
    a, b = a > 0.5, b > 0.5
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0


def refine_with_reference(bad_mask, ref_mask):
    """Affine-fit (translation, rotation, scale) the reference mask (longest-TI image)
    onto a faulty mask, maximising overlap. Returns the warped reference."""
    ref = ref_mask.astype(float)
    if bad_mask.sum() == 0:
        return ref_mask.astype(np.uint8)
    c_ref = np.array(ndi.center_of_mass(ref))
    c_bad = np.array(ndi.center_of_mass(bad_mask))
    x0 = np.array([*(c_bad - c_ref), 0.0, 0.5 * np.log(bad_mask.sum() / ref.sum())])

    def loss(p):
        w = _warp(ref, p, c_ref)
        return 1 - 2 * (w * bad_mask).sum() / (w.sum() + bad_mask.sum() + 1e-9)  # soft dice

    res = minimize(loss, x0, method="Powell", options=dict(maxiter=2000, xtol=1e-2, ftol=1e-4))
    return (_warp(ref, res.x, c_ref) > 0.5).astype(np.uint8)


# ---- pipeline steps: f(masks, ok, tis, pixel_spacing) -> (masks, ok) --------------------------

@POSTPROCESS.register("largest_component")
def step_largest_component(min_area_cm2=MIN_AREA_CM2):
    def f(masks, ok, tis, pixel_spacing):
        masks = np.stack([clean_mask(m, pixel_spacing, min_area_cm2) for m in masks])
        return masks, ok & (masks.reshape(len(masks), -1).sum(1) > 0)
    return f


@POSTPROCESS.register("shape_qc")
def step_shape_qc(max_eccentricity=MAX_ECCENTRICITY):
    def f(masks, ok, tis, pixel_spacing):
        return masks, ok & np.array([shape_ok(m, max_eccentricity) for m in masks])
    return f


@POSTPROCESS.register("refine_from_reference")
def step_refine(reference="longest_ti", max_eccentricity=MAX_ECCENTRICITY):
    """Paper: replace failed masks with the affine-fitted longest-TI mask."""
    def f(masks, ok, tis, pixel_spacing):
        masks, ok = masks.copy(), ok.copy()
        k_ref = int(np.argmax(tis)) if reference == "longest_ti" else int(np.argmin(tis))
        if not ok[k_ref]:
            return masks, ok
        for k in np.where(~ok)[0]:
            refined = refine_with_reference(masks[k], masks[k_ref])
            if shape_ok(refined, max_eccentricity):
                masks[k], ok[k] = refined, True
        return masks, ok
    return f


PAPER_STEPS = ["largest_component", "shape_qc", "refine_from_reference"]


def build_steps(cfg):
    return [POSTPROCESS.build(c) for c in (PAPER_STEPS if cfg is None else cfg)]


def postprocess_series(masks, tis, pixel_spacing=(1.4, 1.4), steps=None):
    """Run post-processing steps (default: the paper's chain). Returns (masks, ok)."""
    steps = build_steps(None) if steps is None else steps
    ok = np.ones(len(masks), bool)
    masks = np.asarray(masks, np.uint8)
    for f in steps:
        masks, ok = f(masks, ok, np.asarray(tis, float), pixel_spacing)
    return masks, ok
