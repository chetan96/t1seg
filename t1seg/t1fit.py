"""Vectorised pixel-wise T1 fitting on magnitude data.

For a fixed T1 the models are linear in the remaining parameters, so we
grid-search T1 and solve the linear part in closed form for every pixel at once
(one shared pseudo-inverse per grid point), then refine with a parabolic step.

  * '2p'  : S = |A (1 - 2 exp(-TI/T1))|                   (STONE / SASHA-like, paper)
  * '3p'  : S = |A - B exp(-TI/T1*)|,  T1 = T1* (B/A - 1)  (MOLLI, Look-Locker corrected)
            sign of the earliest samples is unknown -> polarity restoration by trying
            every flip point and keeping the lowest residual.
"""
import numpy as np

from .registry import FITTERS

T1_GRID = np.arange(50.0, 3000.0, 2.0)


def _parabolic(grid, err, idx):
    i = np.clip(idx, 1, len(grid) - 2)
    n = np.arange(len(i))
    e0, e1, e2 = err[i - 1, n], err[i, n], err[i + 1, n]
    den = e0 - 2 * e1 + e2
    off = np.where(den > 0, 0.5 * (e0 - e2) / np.where(den > 0, den, 1), 0.0)
    return grid[i] + np.clip(off, -1, 1) * (grid[1] - grid[0])


def fit_2p(signal, tis, grid=T1_GRID):
    """signal (K, N) -> T1 (N,), A (N,), rmse (N,)"""
    tis = np.asarray(tis, float)[:, None]
    err = np.empty((len(grid), signal.shape[1]))
    amps = np.empty_like(err)
    for g, t1 in enumerate(grid):
        f = np.abs(1 - 2 * np.exp(-tis / t1))           # (K,1)
        a = (f * signal).sum(0) / (f * f).sum()
        amps[g] = a
        err[g] = ((signal - f * a) ** 2).mean(0)
    idx = err.argmin(0)
    t1 = _parabolic(grid, err, idx)
    n = np.arange(signal.shape[1])
    return t1, amps[idx, n], np.sqrt(err[idx, n])


def fit_3p(signal, tis, grid=T1_GRID):
    """MOLLI 3-parameter fit with polarity restoration. signal (K, N), tis sorted ascending."""
    tis = np.asarray(tis, float)
    order = np.argsort(tis)
    tis, signal = tis[order], signal[order]
    K, N = signal.shape
    best_err = np.full(N, np.inf)
    best = np.zeros((3, N))  # T1*, A, B
    Xs = [np.stack([np.ones(K), -np.exp(-tis / t1s)], 1) for t1s in grid]   # (K,2) each
    pinvs = [np.linalg.pinv(X) for X in Xs]
    for flip in range(K + 1):   # first `flip` samples are negative
        sgn = np.ones((K, 1))
        sgn[:flip] = -1
        y = sgn * signal
        err = np.empty((len(grid), N))
        coef = np.empty((len(grid), 2, N))
        for g, (X, P) in enumerate(zip(Xs, pinvs)):
            c = P @ y                                             # (2,N)
            coef[g] = c
            err[g] = ((y - X @ c) ** 2).mean(0)
        idx = err.argmin(0)
        e = err[idx, np.arange(N)]
        better = e < best_err
        best_err = np.where(better, e, best_err)
        t1s = _parabolic(grid, err, idx)
        best[:, better] = np.stack([t1s, coef[idx, 0, np.arange(N)], coef[idx, 1, np.arange(N)]])[:, better]
    t1s, a, b = best
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = t1s * (b / a - 1)
    return t1, a, np.sqrt(best_err)


def fit_map(stack, tis, fn):
    """Apply a pixel-wise fit fn(signal (K,N), tis) -> T1 (N,) to stack (K, ...). Pixels with a NaN
    in any sample are left NaN."""
    shape = stack.shape[1:]
    s = stack.reshape(stack.shape[0], -1)
    good = np.isfinite(s).all(0)
    t1 = np.full(s.shape[1], np.nan)
    if good.any():
        t1[good] = fn(s[:, good], tis)
    return t1.reshape(shape)


# ---- registered fitters: callable(stack (K, ...), tis) -> T1 map (...) -------------------------

class PixelwiseFitter:
    def __init__(self, t1_min=50.0, t1_max=3000.0, t1_step=2.0):
        self.grid = np.arange(t1_min, t1_max, t1_step)

    def __call__(self, stack, tis):
        return fit_map(stack, tis, lambda s, t: self.fit(s, t, self.grid)[0])


@FITTERS.register("2p")
class TwoParam(PixelwiseFitter):
    """|A (1 - 2 exp(-TI/T1))| -- STONE (paper)."""
    fit = staticmethod(fit_2p)


@FITTERS.register("3p")
class ThreeParamMOLLI(PixelwiseFitter):
    """|A - B exp(-TI/T1*)|, Look-Locker corrected -- MOLLI / shMOLLI."""
    fit = staticmethod(fit_3p)
