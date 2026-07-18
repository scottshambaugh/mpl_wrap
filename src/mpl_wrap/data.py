"""Pure data processing for wrapping: fold arrays into a (min, max) window.

Numeric arrays in, numeric arrays out - no ``Axes`` and no artists. `wrap_line`
and `wrap_points` are the public entry points. The plotting helpers in
``mpl_wrap.plot`` consume them after resolving axis units and stored windows.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
from matplotlib.path import Path

__all__ = [
    "wrap_line",
    "wrap_points",
]


def _wrap_polyline(
    x: np.ndarray, y: np.ndarray, wrapy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a polyline's y into the window as one NaN-broken polyline.

    At each period boundary a segment crosses, the line is routed to the window
    edge, broken with a NaN, and resumed from the opposite edge. Seam crossings
    connect at the correct slope, a multi-period segment sweeps the window once per
    period, and non-finite inputs pass through as breaks.
    To wrap x instead, call with x and y swapped. To wrap both, compose the two.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y0, y1 = wrapy
    period = y1 - y0
    n = len(y)
    if n == 0:
        return x.copy(), y.copy()

    # Each sample's period band is k = floor((y - y0)/period), so its wrapped value
    # is y - k*period, and a segment crosses |dk| period boundaries. Both are computed
    # vectorized, then the sample points and the 3-vertex crossing routes (edge, NaN
    # break, opposite edge) are scattered into one output array by index arithmetic.
    finite = np.isfinite(x) & np.isfinite(y)
    k = np.floor(np.where(finite, (y - y0) / period, 0.0))
    wrapped = np.where(finite, y - k * period, y)

    dk = np.where(finite[:-1] & finite[1:], (k[1:] - k[:-1]), 0.0).astype(np.int64)
    ncross = np.abs(dk)
    total = int(ncross.sum())
    if total == 0:
        return x, wrapped

    before = np.concatenate([[0], np.cumsum(ncross)])  # crossings before each sample
    sample_idx = np.arange(n) + 3 * before
    out_x = np.empty(n + 3 * total)
    out_y = np.empty_like(out_x)
    out_x[sample_idx] = x
    out_y[sample_idx] = wrapped

    seg = np.repeat(np.arange(n - 1), ncross)  # segment each crossing belongs to
    rank = np.arange(total) - np.repeat(before[:-1], ncross)  # 0-based rank within segment
    asc = dk[seg] > 0
    # Ascending crossings go up through boundaries k+1..k[i+1], descending down
    # through k..k[i+1]+1. Interpolate the crossing x on the segment.
    level = np.where(asc, k[:-1][seg] + 1 + rank, k[:-1][seg] - rank)
    yc = y0 + level * period
    xi, xj, yi, yj = x[:-1][seg], x[1:][seg], y[:-1][seg], y[1:][seg]
    xc = xi + (yc - yi) / (yj - yi) * (xj - xi)

    start = sample_idx[:-1][seg] + 1 + 3 * rank  # first of the crossing's 3 vertices
    out_x[start] = out_x[start + 1] = out_x[start + 2] = xc
    out_y[start] = np.where(asc, y1, y0)  # exit edge
    out_y[start + 1] = np.nan
    out_y[start + 2] = np.where(asc, y0, y1)  # enter edge
    return out_x, out_y


def wrap_line(
    x: Any,
    y: Any,
    *,
    wrapx: Iterable[float] | None = None,
    wrapy: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a polyline into the given window(s), without plotting.

    Pure data processing: fold a continuous (unwrapped) polyline into the
    window(s), routing the line to the window edges and inserting a NaN break at
    each seam crossing. The returned arrays are ready for ``ax.plot``. Data and
    windows are plain numbers - for datetime support, use the axes-level helpers
    (or convert with ``matplotlib.dates.date2num`` first).

    Parameters
    ----------
    x, y : array-like
        Continuous (unwrapped) numeric data coordinates.
    wrapx, wrapy : (min, max), optional
        Wrap window per axis. None leaves that axis unwrapped.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The wrapped x and y coordinates, NaN-broken at seam crossings.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if wrapy is not None:
        x, y = _wrap_polyline(x, y, np.asarray(wrapy, dtype=float))
    if wrapx is not None:
        y, x = _wrap_polyline(y, x, np.asarray(wrapx, dtype=float))
    return x, y


def _wrap_points(v: np.ndarray, wrap: np.ndarray | None) -> np.ndarray:
    """Fold point values into the wrap window (NaNs pass through)."""
    return v if wrap is None else (v - wrap[0]) % (wrap[1] - wrap[0]) + wrap[0]


def wrap_points(
    x: Any,
    y: Any,
    *,
    wrapx: Iterable[float] | None = None,
    wrapy: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold points independently into the given window(s), without plotting.

    Pure data processing: each point is folded into the window (a point exactly
    at the window maximum maps to the minimum), and NaNs pass through. Data and
    windows are plain numbers - for datetime support, use the axes-level helpers
    (or convert with ``matplotlib.dates.date2num`` first).

    Parameters
    ----------
    x, y : array-like
        Numeric data coordinates (continuous or already wrapped - folding is
        pointwise).
    wrapx, wrapy : (min, max), optional
        Wrap window per axis. None leaves that axis unwrapped.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The folded x and y coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    wx = None if wrapx is None else np.asarray(wrapx, dtype=float)
    wy = None if wrapy is None else np.asarray(wrapy, dtype=float)
    return _wrap_points(x, wx), _wrap_points(y, wy)


def _contiguous_runs(idx: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into runs of consecutive indices."""
    if len(idx) == 0:
        return []
    return np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)


def _wrap_to_segments(
    x: np.ndarray, y: np.ndarray, wrapx: np.ndarray | None, wrapy: np.ndarray | None
) -> list[np.ndarray]:
    """Wrap a (NaN-broken) polyline and split it into finite runs for a LineCollection."""
    xs, ys = wrap_line(x, y, wrapx=wrapx, wrapy=wrapy)
    idx = np.nonzero(np.isfinite(xs) & np.isfinite(ys))[0]
    return [np.column_stack([xs[run], ys[run]]) for run in _contiguous_runs(idx) if len(run) >= 2]


def _period_and_offsets(wrapy: np.ndarray, *ys: np.ndarray) -> tuple[float, range]:
    """Return the wrap period and the integer period offsets covering the data."""
    y0, y1 = wrapy
    period = y1 - y0
    ymin = min(float(np.nanmin(y)) for y in ys)
    ymax = max(float(np.nanmax(y)) for y in ys)
    m_min = int(np.floor((ymin - y1) / period))
    m_max = int(np.ceil((ymax - y0) / period))
    return period, range(m_min, m_max + 1)


def _tiled_band_vertices(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    wrapx: np.ndarray | None,
    wrapy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile the band at every (x, y) period offset into one path (vectorized).

    Built with numpy broadcasting rather than a nested per-offset Python loop, so
    a fast angle needing thousands of tiles is built in one shot.
    """
    px = 0.0 if wrapx is None else wrapx[1] - wrapx[0]
    py = 0.0 if wrapy is None else wrapy[1] - wrapy[0]
    x_offsets = range(1) if wrapx is None else _period_and_offsets(wrapx, x)[1]
    y_offsets = range(1) if wrapy is None else _period_and_offsets(wrapy, lo, hi)[1]
    my = np.fromiter(y_offsets, dtype=float)
    nx = np.fromiter(x_offsets, dtype=float)
    m_flat = np.repeat(my, len(nx))
    n_flat = np.tile(nx, len(my))
    tile_len = 2 * len(x)
    x_tile = np.concatenate([x, x[::-1]])
    y_tile = np.concatenate([lo, hi[::-1]])
    x_all = (x_tile[None, :] - n_flat[:, None] * px).ravel()
    y_all = (y_tile[None, :] - m_flat[:, None] * py).ravel()
    codes = np.full(len(m_flat) * tile_len, Path.LINETO, dtype=np.uint8)
    codes[::tile_len] = Path.MOVETO
    return np.column_stack([x_all, y_all]), codes


def _saturated_band_vertices(
    x: np.ndarray, lo: np.ndarray, hi: np.ndarray, full: np.ndarray, wrapy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the y-wrapped band path, collapsing saturated x-runs to a rectangle.

    Where the band spans a full period it fills the window, so each contiguous
    saturated x-run becomes one window-height rectangle instead of a stack of
    tiles. The remaining narrow runs are tiled locally over just their own period
    offsets. This avoids building the thousands of window-spanning tiles a fast,
    mostly-saturated angle would otherwise need.
    """
    y0, y1 = wrapy
    period = y1 - y0
    verts: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    rect_codes = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO], dtype=np.uint8)
    for run in _contiguous_runs(np.nonzero(full)[0]):
        xa, xb = x[run[0]], x[run[-1]]
        verts.append(np.array([[xa, y0], [xb, y0], [xb, y1], [xa, y1]]))
        codes.append(rect_codes)
    n = len(x)
    for run in _contiguous_runs(np.nonzero(~full)[0]):
        # Extend each narrow run by one sample into the adjacent saturated region so
        # its tiles bridge the transition and abut the rectangle (they meet at the
        # boundary x-line, so no gap and no overlapping-alpha).
        sl = slice(max(int(run[0]) - 1, 0), min(int(run[-1]) + 1, n - 1) + 1)
        lo_r, hi_r, x_r = lo[sl], hi[sl], x[sl]
        m = np.arange(
            int(np.floor((lo_r.min() - y1) / period)),
            int(np.ceil((hi_r.max() - y0) / period)) + 1,
            dtype=float,
        )
        tile_len = 2 * len(x_r)
        x_tile = np.tile(np.concatenate([x_r, x_r[::-1]]), len(m))
        y_tile = (np.concatenate([lo_r, hi_r[::-1]])[None, :] - m[:, None] * period).ravel()
        verts.append(np.column_stack([x_tile, y_tile]))
        c = np.full(len(m) * tile_len, Path.LINETO, dtype=np.uint8)
        c[::tile_len] = Path.MOVETO
        codes.append(c)
    if not verts:
        return np.empty((0, 2)), np.empty(0, dtype=np.uint8)
    return np.concatenate(verts), np.concatenate(codes)


def _error_bounds(values: np.ndarray, error: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper bounds for symmetric or asymmetric errors."""
    e = np.asarray(error, dtype=float)
    lower = e[0] if e.ndim == 2 else e
    upper = e[1] if e.ndim == 2 else e
    return values - lower, values + upper


def _nan_joined_extents(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Join error-bar extents into one NaN-separated coordinate array."""
    out = np.empty(3 * len(lo))
    out[0::3], out[1::3], out[2::3] = lo, hi, np.nan
    return out
