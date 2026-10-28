"""Polar resampling of the myocardium (paper: 'T1 map reconstruction and analysis').

Each image's myocardium is sampled on 360 rays (1 deg) from its own mask's center
of mass, with C points between endo- and epicardium. Doing this per image aligns
the myocardium across inversion times without image registration.
"""
import numpy as np
from scipy import ndimage as ndi

N_ANGLES = 360
C_DEPTH = 20  # > max wall thickness in pixels (paper: 15), avoids many-to-one sampling


def ray_borders(mask, center, n_angles=N_ANGLES, step=0.25):
    """Endo/epi radius along each ray (NaN where the ray misses the myocardium)."""
    h, w = mask.shape
    r = np.arange(0, np.hypot(h, w) / 2, step)
    th = np.deg2rad(np.arange(n_angles) * 360.0 / n_angles)
    ys = center[0] + np.outer(np.sin(th), r)
    xs = center[1] + np.outer(np.cos(th), r)
    hit = ndi.map_coordinates(mask.astype(float), [ys, xs], order=1, cval=0) > 0.5
    any_hit = hit.any(1)
    first = np.argmax(hit, 1)
    last = hit.shape[1] - 1 - np.argmax(hit[:, ::-1], 1)
    r_in = np.where(any_hit, r[first], np.nan)
    r_out = np.where(any_hit, r[last], np.nan)
    return r_in, r_out, th


def to_polar(image, mask, n_angles=N_ANGLES, depth=C_DEPTH):
    """Returns polar image (depth, n_angles) plus the geometry used."""
    center = np.array(ndi.center_of_mass(mask > 0))
    r_in, r_out, th = ray_borders(mask, center, n_angles)
    u = np.linspace(0, 1, depth)[:, None]
    rr = r_in[None] + u * (r_out - r_in)[None]          # (depth, n_angles)
    ys = center[0] + rr * np.sin(th)[None]
    xs = center[1] + rr * np.cos(th)[None]
    valid = np.isfinite(rr)
    p = ndi.map_coordinates(image.astype(float), [np.nan_to_num(ys), np.nan_to_num(xs)], order=1)
    p[~valid] = np.nan
    return p, dict(center=center, r_in=r_in, r_out=r_out)


def from_polar(polar_map, ref_mask, geom=None):
    """Inverse transform a (depth, n_angles) map onto the Cartesian grid of ref_mask."""
    depth, n_angles = polar_map.shape
    if geom is None:
        _, geom = to_polar(ref_mask.astype(float), ref_mask, n_angles, depth)
    cy, cx = geom["center"]
    yy, xx = np.nonzero(ref_mask)
    ang = np.mod(np.degrees(np.arctan2(yy - cy, xx - cx)), 360.0) * n_angles / 360.0
    rad = np.hypot(yy - cy, xx - cx)
    # endo/epi radius at each pixel's angle (linear interp, wrapping around 360)
    a_idx = np.arange(n_angles + 1)
    r_in = np.interp(ang, a_idx, np.append(geom["r_in"], geom["r_in"][0]))
    r_out = np.interp(ang, a_idx, np.append(geom["r_out"], geom["r_out"][0]))
    u = np.clip((rad - r_in) / np.maximum(r_out - r_in, 1e-6), 0, 1) * (depth - 1)
    # pad one wrapped column so bilinear interp works across 359 -> 0 deg
    padded = np.concatenate([polar_map, polar_map[:, :1]], axis=1)
    vals = ndi.map_coordinates(np.nan_to_num(padded, nan=0.0), [u, ang], order=1)
    nanmask = ndi.map_coordinates(np.isnan(padded).astype(float), [u, ang], order=1) > 0.5
    out = np.full(ref_mask.shape, np.nan)
    out[yy, xx] = np.where(nanmask, np.nan, vals)
    return out
