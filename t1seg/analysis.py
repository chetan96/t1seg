"""Mid-wall pruning and T1 statistics (paper: border exclusion + 850-1500 ms acceptance range)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk, skeletonize

T1_RANGE_1P5T = (850.0, 1500.0)  # native myocardium at 1.5 T; change for 3 T / post-contrast


def midwall_mask(mask, fraction=1 / 3):
    """Keep a band around the myocardial skeleton whose width is `fraction` of mean wall thickness,
    excluding sub-endo/epicardial border pixels (partial volume with blood / fat)."""
    mask = mask > 0
    skel = skeletonize(mask)
    if skel.sum() == 0:
        return mask
    thickness = mask.sum() / skel.sum()          # area / centerline length
    radius = max(0, int(round(fraction * thickness / 2)))
    band = ndi.binary_dilation(skel, disk(radius)) if radius > 0 else skel
    return band & mask


def slice_stats(t1_map, mask, t1_range=T1_RANGE_1P5T):
    v = t1_map[(mask > 0) & np.isfinite(t1_map)]
    n_all = v.size
    v = v[(v >= t1_range[0]) & (v <= t1_range[1])]
    return dict(mean=float(v.mean()) if v.size else np.nan,
                sd=float(v.std()) if v.size else np.nan,
                n=int(v.size), n_rejected=int(n_all - v.size))


def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0
