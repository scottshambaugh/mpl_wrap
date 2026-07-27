"""Pure data processing for wrapping: fold arrays into a (min, max) window.

Numeric arrays in, numeric arrays out - no ``Axes`` and no artists. `wrap_line`
and `wrap_points` are the public entry points. The plotting helpers in
``mpl_wrap.plot`` consume them after resolving axis units and stored windows.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
from matplotlib.path import Path
from matplotlib.transforms import Bbox

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


def _step_polyline(x: np.ndarray, *ys: np.ndarray, where: str) -> tuple[np.ndarray, ...]:
    """Expand a step series into the tread/riser polyline that ``ax.step`` draws.

    ``where`` places the risers as matplotlib does: "pre" holds each value over
    ``(x[i-1], x[i]]``, "post" over ``[x[i], x[i+1])``, and "mid" steps half-way
    between x positions. Several y arrays can be stepped against one x, as in
    ``ax.fill_between(..., step=...)``.
    """
    if where == "pre":
        return (np.repeat(x, 2)[:-1], *(np.repeat(y, 2)[1:] for y in ys))
    if where == "post":
        return (np.repeat(x, 2)[1:], *(np.repeat(y, 2)[:-1] for y in ys))
    if where == "mid":
        mids = 0.5 * (x[:-1] + x[1:])
        return (np.concatenate([x[:1], np.repeat(mids, 2), x[-1:]]), *(np.repeat(y, 2) for y in ys))
    raise ValueError(f"must be one of ['pre', 'post', 'mid'], not {where!r}")


def _stairs_polyline(
    values: np.ndarray, edges: np.ndarray, baseline: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Expand bin values and edges into the tread/riser polyline that ``ax.stairs`` draws.

    As in ``ax.stairs``, a scalar ``baseline`` drops the two ends down to it, an
    array ``baseline`` closes the path back along its own staircase, and None
    leaves the staircase open.
    """
    x = np.repeat(edges, 2)
    y = np.repeat(values, 2)
    if baseline is None:
        return x[1:-1], y
    if baseline.ndim == 0:
        return x, np.concatenate([[baseline], y, [baseline]])
    base = np.repeat(baseline, 2)[::-1]
    return (
        np.concatenate([x, x[::-1]]),
        np.concatenate([base[-1:], y, base[:1], base[:1], base, base[-1:]]),
    )


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
    return _closed_subpaths(np.column_stack([x_all, y_all]), codes)


def _saturated_band_vertices(
    x: np.ndarray, lo: np.ndarray, hi: np.ndarray, full: np.ndarray, wrapy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the y-wrapped band path, collapsing saturated x-runs to a rectangle.

    Where the band spans a full period it fills the window, so each contiguous
    saturated x-run becomes one rectangle instead of a stack of tiles. The
    remaining narrow runs are tiled locally over just their own period offsets.
    This avoids building the thousands of window-spanning tiles a fast, mostly
    saturated angle would otherwise need.

    The rectangle runs a period past the window on each side, so that an edge,
    if stroked, is clipped away: a saturated stretch is covered everywhere, so
    it has no boundary to draw. Where two pieces meet, the join is not a band
    edge either, so each piece is left open along the side it shares - filling
    closes a subpath implicitly, stroking does not - and the piece is traced
    from that side so the open edge falls there.
    """
    y0, y1 = wrapy
    period = y1 - y0
    verts: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    n = len(x)
    bounds = np.concatenate([[0], np.nonzero(np.diff(full))[0] + 1, [n]])
    for start, stop in zip(bounds[:-1], bounds[1:]):
        # An edge shared with the neighbouring piece is a join, not a band edge.
        open_left, open_right = start > 0, stop < n
        if full[start]:
            xa, xb = x[start], x[stop - 1]
            ya, yb = y0 - period, y1 + period
            # Rolled (not reversed) to move the open edge: reversing would flip the
            # winding, and an opposite-wound piece cancels where it meets its neighbour.
            rect = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
            verts.append(np.roll(rect, -2, axis=0) if open_right and not open_left else rect)
            codes.append(np.array([Path.MOVETO, *[Path.LINETO] * 3], dtype=np.uint8))
        else:
            # Extend each narrow run by one sample into the adjacent saturated region
            # so its tiles bridge the transition and abut the rectangle (they meet at
            # the boundary x-line, so no gap and no overlapping-alpha).
            sl = slice(max(int(start) - 1, 0), min(int(stop - 1) + 1, n - 1) + 1)
            lo_r, hi_r, x_r = lo[sl], hi[sl], x[sl]
            m = np.arange(
                int(np.floor((lo_r.min() - y1) / period)),
                int(np.ceil((hi_r.max() - y0) / period)) + 1,
                dtype=float,
            )
            # The ring closes along its left cap; rolling its start by one side's
            # worth of samples moves that open edge to the right cap instead.
            x_ring = np.concatenate([x_r, x_r[::-1]])
            y_ring = np.concatenate([lo_r, hi_r[::-1]])
            if open_right and not open_left:
                x_ring, y_ring = np.roll(x_ring, -len(x_r)), np.roll(y_ring, -len(x_r))
            tile_len = 2 * len(x_r)
            x_tile = np.tile(x_ring, len(m))
            y_tile = (y_ring[None, :] - m[:, None] * period).ravel()
            verts.append(np.column_stack([x_tile, y_tile]))
            c = np.full(len(m) * tile_len, Path.LINETO, dtype=np.uint8)
            c[::tile_len] = Path.MOVETO
            codes.append(c)
        if not (open_left or open_right):  # both caps are real: close it, as ax.fill_between does
            verts[-1], codes[-1] = _closed_subpaths(verts[-1], codes[-1])
    if not verts:
        return np.empty((0, 2)), np.empty(0, dtype=np.uint8)
    return np.concatenate(verts), np.concatenate(codes)


def _interp_crossing(
    x: np.ndarray, y1: np.ndarray, y2: np.ndarray, idx: int
) -> tuple[float, float]:
    """Interpolate where the two curves cross, as ``fill_between(interpolate=True)`` does.

    Solves ``y1 - y2 == 0`` on the segment ending at ``idx`` and evaluates y
    there, clamping to the segment ends when the curves do not actually cross.
    """
    sl = slice(max(idx - 1, 0), idx + 1)
    xs, diff = x[sl], y1[sl] - y2[sl]
    order = np.argsort(diff)
    xc = float(np.interp(0.0, diff[order], xs[order]))
    order = np.argsort(xs)
    return xc, float(np.interp(xc, xs[order], y1[sl][order]))


def _run_band_vertices(
    x: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    wrapx: np.ndarray | None,
    wrapy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the wrapped path for one contiguous band run."""
    # Clamp bands to one y-period and record where they saturate (fill the window).
    if wrapy is not None:
        period = wrapy[1] - wrapy[0]
        full = (hi - lo) >= period
        hi = lo + np.minimum(hi - lo, period)
    else:
        full = np.zeros(len(x), dtype=bool)

    # y-only wrap: saturated x-runs collapse to one rectangle each, else tile in x/y.
    if wrapx is None and wrapy is not None:
        return _saturated_band_vertices(x, lo, hi, full, wrapy)
    return _tiled_band_vertices(x, lo, hi, wrapx, wrapy)


def _band_vertices(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    *,
    where: Any = None,
    interpolate: bool = False,
    step: str | None = None,
    wrapx: np.ndarray | None = None,
    wrapy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the wrapped fill-band path, with ``ax.fill_between``'s run semantics.

    The band is drawn once per contiguous run of ``where`` (non-finite samples
    break runs, as gappy series are the norm for wrapped data), each run stepped
    and interpolated as requested before being tiled over its period offsets.
    """
    x, y1, y2 = np.broadcast_arrays(np.atleast_1d(x), np.atleast_1d(y1), np.atleast_1d(y2))
    mask = np.isfinite(x) & np.isfinite(y1) & np.isfinite(y2)
    if where is not None:
        where = np.asarray(where, dtype=bool)
        if where.size != x.size:
            raise ValueError(f"where size ({where.size}) does not match x size ({x.size})")
        mask &= where

    n = len(x)
    verts: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for run in _contiguous_runs(np.nonzero(mask)[0]):
        idx0, idx1 = int(run[0]), int(run[-1]) + 1
        xr, f1, f2 = x[idx0:idx1], y1[idx0:idx1], y2[idx0:idx1]
        if step is not None:
            xr, f1, f2 = _step_polyline(xr, f1, f2, where=step)
        lo, hi = np.minimum(f1, f2), np.maximum(f1, f2)
        # Extend the run to the true crossing points, pinching the band shut there.
        # Only inside the data: at the array ends there is no segment to cross on.
        if interpolate and idx0 > 0:
            xc, yc = _interp_crossing(x, y1, y2, idx0)
            xr, lo, hi = (np.concatenate([[v], a]) for v, a in ((xc, xr), (yc, lo), (yc, hi)))
        if interpolate and idx1 < n:
            xc, yc = _interp_crossing(x, y1, y2, idx1)
            xr, lo, hi = (np.concatenate([a, [v]]) for v, a in ((xc, xr), (yc, lo), (yc, hi)))
        v, c = _run_band_vertices(xr, lo, hi, wrapx, wrapy)
        verts.append(v)
        codes.append(c)
    if not verts:
        return np.empty((0, 2)), np.empty(0, dtype=np.uint8)
    return np.concatenate(verts), np.concatenate(codes)


def _polyline_path(x: np.ndarray, y: np.ndarray) -> Path:
    """Turn a NaN-broken polyline into a Path, one subpath per finite run."""
    runs = [
        r for r in _contiguous_runs(np.nonzero(np.isfinite(x) & np.isfinite(y))[0]) if len(r) > 1
    ]
    if not runs:
        return Path(np.empty((0, 2)), np.empty(0, dtype=np.uint8))
    verts = np.concatenate([np.column_stack([x[r], y[r]]) for r in runs])
    codes = np.full(len(verts), Path.LINETO, dtype=np.uint8)
    codes[np.cumsum([0] + [len(r) for r in runs[:-1]])] = Path.MOVETO
    return Path(verts, codes)


def _closed_subpaths(verts: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Terminate every subpath with an explicit CLOSEPOLY.

    Filling closes subpaths implicitly, but point-in-path tests do not: without
    this, a hit test on the compound path joins one tile to the next and reports
    coverage between them.
    """
    if len(verts) == 0:
        return verts, codes
    starts = np.nonzero(codes == Path.MOVETO)[0]
    ends = np.append(starts[1:], len(codes))
    close = np.array([Path.CLOSEPOLY], dtype=np.uint8)
    out_v = [a for s, e in zip(starts, ends) for a in (verts[s:e], verts[s : s + 1])]
    out_c = [a for s, e in zip(starts, ends) for a in (codes[s:e], close)]
    return np.concatenate(out_v), np.concatenate(out_c)


def _band_extent(verts: np.ndarray, wrapx: np.ndarray | None, wrapy: np.ndarray | None) -> Bbox:
    """The extent a tiled band actually covers: the window where wrapped, its own where not.

    The tiling deliberately overshoots the window, so the raw vertices span many
    periods; what is drawn is the intersection with the window. A band narrower
    than the period leaves gaps inside the window, which this does not subtract.
    """
    if len(verts) == 0:
        return Bbox.null()
    x0, x1 = wrapx if wrapx is not None else (verts[:, 0].min(), verts[:, 0].max())
    y0, y1 = wrapy if wrapy is not None else (verts[:, 1].min(), verts[:, 1].max())
    return Bbox.from_extents(float(x0), float(y0), float(x1), float(y1))


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
