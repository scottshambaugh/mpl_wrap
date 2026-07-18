"""Helper functions for plotting wrapped, angular, or periodic data on matplotlib axes.

Free functions that mirror the core matplotlib plotting methods, taking an
``Axes`` plus ``wrapx`` / ``wrapy`` (min, max) windows: continuous (unwrapped)
data is folded into the window, with lines routed to the window edges at each
seam crossing instead of drawing jump artifacts. ``set_wrap`` stores a window
on an axes so subsequent calls pick it up automatically.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, Union, overload

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

__all__ = [
    "set_wrap",
    "plot_wrapped",
    "scatter_wrapped",
    "fill_between_wrapped",
    "stairs_wrapped",
    "errorbar_wrapped",
]

# A wrap window spec: a (min, max) pair in data units (datetimes allowed), False
# to explicitly disable wrapping on an axis with a stored window, or None to fall
# back to the window stored by set_wrap (if any).
WrapSpec = Union[Iterable[Any], Literal[False], None]

_WINDOW_ATTR = "_mpl_wrap_windows"


@overload
def _to_num(axis: Axis, values: None) -> None: ...
@overload
def _to_num(axis: Axis, values: Any) -> np.ndarray: ...


def _to_num(axis: Axis, values: Any) -> np.ndarray | None:
    """Register units on the axis and return values in matplotlib's numeric form.

    Lets datetime (and other unit-ful) inputs be wrapped: the axis learns the
    converter, so ticks still format correctly, and we get plain floats to do the
    wrapping arithmetic on. Passes None and already-numeric data straight through.
    """
    if values is None:
        return None

    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float)

    # Register the converter (and its tick locators/formatters) once per axis
    if axis.get_converter() is None:
        axis.update_units(values)
    return np.asarray(axis.convert_units(values), dtype=float)


def _resolve_wrap(ax: Axes, name: str, wrap: WrapSpec) -> np.ndarray | None:
    """Resolve a wrap spec: explicit window, else the set_wrap stored window, else None.

    False explicitly disables wrapping even when a stored window exists.
    """
    if wrap is False:
        return None
    axis = ax.xaxis if name == "x" else ax.yaxis
    if wrap is not None:
        return _to_num(axis, wrap)
    stored: dict[str, np.ndarray] = getattr(ax, _WINDOW_ATTR, {})
    return stored.get(name)


def _period_and_offsets(wrapy: np.ndarray, *ys: np.ndarray) -> tuple[float, range]:
    """Return the wrap period and the integer period offsets covering the data."""
    y0, y1 = wrapy
    period = y1 - y0
    ymin = min(float(np.nanmin(y)) for y in ys)
    ymax = max(float(np.nanmax(y)) for y in ys)
    m_min = int(np.floor((ymin - y1) / period))
    m_max = int(np.ceil((ymax - y0) / period))
    return period, range(m_min, m_max + 1)


def _wrap_polyline(
    x: np.ndarray, y: np.ndarray, wrapy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a polyline's y into the window as one NaN-broken polyline.

    At each period boundary a segment crosses, the line is routed to the window
    edge, broken with a NaN, and resumed from the opposite edge. Seam crossings
    connect at the correct slope, a multi-period segment sweeps the window once per
    period, and non-finite inputs pass through as breaks - one artist, no clipping.
    To wrap x instead, call with x and y swapped; to wrap both, compose the two.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y0, y1 = wrapy
    period = y1 - y0
    n = len(y)
    if n == 0:
        return x.copy(), y.copy()

    # Each sample's period band is k = floor((y - y0)/period), so its wrapped value
    # is y - k*period; a segment crosses |dk| period boundaries. Both are computed
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
    # Ascending crossings go up through boundaries k+1..k[i+1]; descending down
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


def _wrap_xy(
    x: np.ndarray,
    y: np.ndarray,
    wrapx: np.ndarray | None,
    wrapy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap a polyline into the given x and/or y windows by composing _wrap_polyline."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if wrapy is not None:
        x, y = _wrap_polyline(x, y, wrapy)
    if wrapx is not None:
        y, x = _wrap_polyline(y, x, wrapx)
    return x, y


def _wrap_points(v: np.ndarray, wrap: np.ndarray | None) -> np.ndarray:
    """Fold point values into the wrap window (NaNs pass through)."""
    return v if wrap is None else (v - wrap[0]) % (wrap[1] - wrap[0]) + wrap[0]


def _clip_patch_to_window(
    ax: Axes,
    patch: PathPatch,
    wrapx: np.ndarray | None,
    wrapy: np.ndarray | None,
) -> None:
    """Clip a filled patch to the wrap window(s); a wrapped axis in data, the other full."""
    if wrapx is not None and wrapy is not None:
        rect = Rectangle(
            (wrapx[0], wrapy[0]), wrapx[1] - wrapx[0], wrapy[1] - wrapy[0], transform=ax.transData
        )
    elif wrapy is not None:
        rect = Rectangle(
            (0.0, wrapy[0]), 1.0, wrapy[1] - wrapy[0], transform=ax.get_yaxis_transform()
        )
    elif wrapx is not None:
        rect = Rectangle(
            (wrapx[0], 0.0), wrapx[1] - wrapx[0], 1.0, transform=ax.get_xaxis_transform()
        )
    else:
        return
    patch.set_clip_path(rect)


def _contiguous_runs(idx: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into runs of consecutive indices."""
    if len(idx) == 0:
        return []
    return np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)


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
    tiles; the remaining narrow runs are tiled locally over just their own period
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


def _wrap_to_segments(
    x: np.ndarray, y: np.ndarray, wrapx: np.ndarray | None, wrapy: np.ndarray | None
) -> list[np.ndarray]:
    """Wrap a (NaN-broken) polyline and split it into finite runs for a LineCollection."""
    xs, ys = _wrap_xy(x, y, wrapx, wrapy)
    idx = np.nonzero(np.isfinite(xs) & np.isfinite(ys))[0]
    if len(idx) == 0:
        return []
    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    return [np.column_stack([xs[r], ys[r]]) for r in runs if len(r) >= 2]


def set_wrap(
    ax: Axes,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    *,
    set_lims: bool = True,
    seam_lines: bool = True,
    margin: float = 0.05,
    seam_kwargs: dict[str, Any] | None = None,
) -> list[Line2D]:
    """Store wrap window(s) on an axes so the plotting helpers use them by default.

    After ``set_wrap(ax, wrapy=(0, 360))``, helpers called on ``ax`` without an
    explicit ``wrapy`` wrap into the stored window; an explicit per-call window
    still overrides, and ``wrapx=False`` / ``wrapy=False`` disables wrapping for
    a single call. Calling ``set_wrap`` again updates only the windows given
    (pass ``False`` to clear a stored window).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window for the x/y axis, in data units (datetimes allowed).
        ``False`` clears a previously stored window; None leaves it unchanged.
    set_lims : bool, default True
        Set the axis limits of each given window to the window plus ``margin``.
    seam_lines : bool, default True
        Draw lines at the window edges of each given window.
    margin : float, default 0.05
        Fraction of the period to pad the axis limits by when ``set_lims``.
    seam_kwargs : dict, optional
        Overrides for the seam line style (default ``color="k", linewidth=0.8``).

    Returns
    -------
    list of matplotlib.lines.Line2D
        The seam line artists that were drawn (empty if ``seam_lines=False``).
    """
    windows: dict[str, np.ndarray] = dict(getattr(ax, _WINDOW_ATTR, {}))
    artists: list[Line2D] = []
    style: dict[str, Any] = {"color": "k", "linewidth": 0.8}
    style.update(seam_kwargs or {})
    for name, wrap, axis, set_lim, seam in (
        ("x", wrapx, ax.xaxis, ax.set_xlim, ax.axvline),
        ("y", wrapy, ax.yaxis, ax.set_ylim, ax.axhline),
    ):
        if wrap is None:
            continue
        if wrap is False:
            windows.pop(name, None)
            continue
        w = _to_num(axis, wrap)
        windows[name] = w
        if set_lims:
            pad = margin * (w[1] - w[0])
            set_lim(w[0] - pad, w[1] + pad)
        if seam_lines:
            artists.append(seam(w[0], **style))
            artists.append(seam(w[1], **style))
    setattr(ax, _WINDOW_ATTR, windows)
    return artists


def plot_wrapped(
    ax: Axes,
    x: Any,
    y: Any,
    *args: Any,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> list[Line2D]:
    """Plot a continuous (unwrapped) series on a wrapped axis.

    Mirrors ``ax.plot`` with optional ``wrapx`` and/or ``wrapy`` (min, max)
    windows. Pass continuous (unwrapped) data - pre-wrapped data should be made
    continuous first (``np.unwrap``). Datetime data and windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x, y : array-like
        Continuous (unwrapped) data coordinates.
    *args, **kwargs
        Forwarded to ``ax.plot`` (format string, styling, ...).
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis; defaults to the window stored by `set_wrap`,
        ``False`` disables wrapping for this call.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The plotted line artists, as from ``ax.plot``.
    """
    x = _to_num(ax.xaxis, x)
    y = _to_num(ax.yaxis, y)
    wx = _resolve_wrap(ax, "x", wrapx)
    wy = _resolve_wrap(ax, "y", wrapy)
    xs, ys = _wrap_xy(x, y, wx, wy)
    return ax.plot(xs, ys, *args, **kwargs)


def scatter_wrapped(
    ax: Axes,
    x: Any,
    y: Any,
    *args: Any,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> PathCollection:
    """Scatter points on a wrapped axis, folding each point into the window.

    Mirrors ``ax.scatter`` with optional ``wrapx`` and/or ``wrapy`` (min, max)
    windows. Each point is independently folded into the window (a point exactly
    at the window maximum maps to the minimum). Datetime data and windows are
    accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x, y : array-like
        Data coordinates (continuous or already wrapped - folding is pointwise).
    *args, **kwargs
        Forwarded to ``ax.scatter`` (sizes, colors, styling, ...).
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis; defaults to the window stored by `set_wrap`,
        ``False`` disables wrapping for this call.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter artist, as from ``ax.scatter``.
    """
    x = _to_num(ax.xaxis, x)
    y = _to_num(ax.yaxis, y)
    wx = _resolve_wrap(ax, "x", wrapx)
    wy = _resolve_wrap(ax, "y", wrapy)
    return ax.scatter(_wrap_points(x, wx), _wrap_points(y, wy), *args, **kwargs)


def fill_between_wrapped(
    ax: Axes,
    x: Any,
    y1: Any,
    y2: Any,
    *,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> PathPatch:
    """Fill between two continuous (unwrapped) series on a wrapped axis.

    Mirrors ``ax.fill_between`` with optional ``wrapx`` and/or ``wrapy``
    (min, max) windows. The band is tiled at every period offset (in x and/or y)
    into one clipped compound path, so the union fills once with no double alpha,
    and a band at least a y-period wide fills the whole window as "fully
    uncertain". The fill is the one helper that clips rather than routing to
    edges. Datetime data and windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x : array-like
        Continuous (unwrapped) x coordinates.
    y1, y2 : array-like
        The band edges (continuous / unwrapped); order does not matter.
    **kwargs
        Forwarded to the ``matplotlib.patches.PathPatch`` (color, alpha, ...).
        ``linewidth`` defaults to 0.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis; defaults to the window stored by `set_wrap`,
        ``False`` disables wrapping for this call.

    Returns
    -------
    matplotlib.patches.PathPatch
        The band artist, added to the axes (excluded from data limits and layout).
    """
    x = _to_num(ax.xaxis, x)
    y1 = _to_num(ax.yaxis, y1)
    y2 = _to_num(ax.yaxis, y2)
    wrapx = _resolve_wrap(ax, "x", wrapx)
    wrapy = _resolve_wrap(ax, "y", wrapy)
    lo = np.minimum(y1, y2)
    hi = np.maximum(y1, y2)

    # Clamp bands to one y-period and record where they saturate (fill the window).
    if wrapy is not None:
        period = wrapy[1] - wrapy[0]
        full = (hi - lo) >= period
        hi = lo + np.minimum(hi - lo, period)
    else:
        full = np.zeros(len(x), dtype=bool)

    # y-only wrap: saturated x-runs collapse to one rectangle each; else tile in x/y.
    if wrapx is None and wrapy is not None:
        verts, codes = _saturated_band_vertices(x, lo, hi, full, wrapy)
    else:
        verts, codes = _tiled_band_vertices(x, lo, hi, wrapx, wrapy)

    kwargs.setdefault("linewidth", 0)
    patch = PathPatch(Path(verts, codes), **kwargs)
    patch.set_transform(ax.transData)
    # Clipped to the window, so keep its huge path out of datalim (add_artist) and
    # layout (set_in_layout) - both would otherwise walk every tiled vertex.
    ax.add_artist(patch)
    _clip_patch_to_window(ax, patch, wrapx, wrapy)
    patch.set_in_layout(False)
    return patch


def stairs_wrapped(
    ax: Axes,
    values: Any,
    edges: Any = None,
    *,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> list[Line2D]:
    """Draw a continuous (unwrapped) step series on a wrapped axis.

    Mirrors ``ax.stairs`` with optional ``wrapx`` and/or ``wrapy`` (min, max)
    windows. The staircase is turned into a tread/riser polyline and wrapped, so
    seam-crossing risers route to the edges. Rendered with ``ax.plot``, so
    stairs-only kwargs (``baseline``, ``fill``) do not apply. Datetime data and
    windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    values : array-like
        Step heights (continuous / unwrapped).
    edges : array-like, optional
        Bin edges, one longer than ``values``; defaults to ``arange(len(values) + 1)``.
    **kwargs
        Forwarded to ``ax.plot`` (``baseline`` and ``fill`` are ignored).
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis; defaults to the window stored by `set_wrap`,
        ``False`` disables wrapping for this call.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The plotted line artists, as from ``ax.plot``.
    """
    values = _to_num(ax.yaxis, values)
    if edges is None:
        edges = np.arange(len(values) + 1, dtype=float)
    else:
        edges = _to_num(ax.xaxis, edges)
    wx = _resolve_wrap(ax, "x", wrapx)
    wy = _resolve_wrap(ax, "y", wrapy)
    kwargs.pop("baseline", None)
    kwargs.pop("fill", None)

    step_x = np.repeat(edges, 2)[1:-1]
    step_y = np.repeat(values, 2)
    return ax.plot(*_wrap_xy(step_x, step_y, wx, wy), **kwargs)


def errorbar_wrapped(
    ax: Axes,
    x: Any,
    y: Any,
    yerr: Any = None,
    xerr: Any = None,
    fmt: str = "",
    *,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    ecolor: Any = None,
    elinewidth: float | None = None,
    capsize: float | None = None,
    **kwargs: Any,
) -> ErrorbarContainer:
    """Draw error bars on a wrapped axis, returning a matplotlib ErrorbarContainer.

    Mirrors ``ax.errorbar`` and its return type - a data line, a caplines tuple,
    and a barlinecols tuple of LineCollections - with optional ``wrapx`` and/or
    ``wrapy`` (min, max) windows. Each bar is wrapped into the window and split
    at the seam, and (as in core errorbar) all bar segments go into one
    LineCollection: a bar straddling the seam shows at both edges and one
    spanning a full period sweeps the window. Datetime data and windows are
    accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x, y : array-like
        Data coordinates (continuous / unwrapped).
    yerr, xerr : array-like, optional
        Symmetric (n,) or asymmetric (2, n) error extents.
    fmt : str, default ""
        Format string for the data markers; ``"none"`` suppresses them.
    ecolor, elinewidth, capsize
        Bar and cap styling, as in ``ax.errorbar``.
    **kwargs
        Forwarded to ``ax.plot`` for the data line/markers.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis; defaults to the window stored by `set_wrap`,
        ``False`` disables wrapping for this call.

    Returns
    -------
    matplotlib.container.ErrorbarContainer
        Container of (data line, caplines, barlinecols), as from ``ax.errorbar``.
    """
    x = _to_num(ax.xaxis, x)
    y = _to_num(ax.yaxis, y)
    wrapx = _resolve_wrap(ax, "x", wrapx)
    wrapy = _resolve_wrap(ax, "y", wrapy)
    label = kwargs.pop("label", None)

    # Data line / markers at the wrapped centres ('none' suppresses them, as in ax.errorbar).
    data_line: Line2D | None = None
    if fmt != "none":
        drawn = ax.plot(_wrap_points(x, wrapx), _wrap_points(y, wrapy), fmt, label=label, **kwargs)
        data_line = drawn[0] if drawn else None

    bar_color = (
        ecolor if ecolor is not None else (data_line.get_color() if data_line is not None else "C0")
    )
    bar_lw = elinewidth if elinewidth is not None else mpl.rcParams["lines.linewidth"]
    barlinecols: list[LineCollection] = []
    caplines: list[Line2D] = []

    def bar_extent(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        out = np.empty(3 * len(lo))
        out[0::3], out[1::3], out[2::3] = lo, hi, np.nan
        return out

    def add_caps(cx: np.ndarray, cy: np.ndarray, marker: str) -> None:
        if capsize:
            (cap,) = ax.plot(
                _wrap_points(cx, wrapx),
                _wrap_points(cy, wrapy),
                linestyle="none",
                marker=marker,
                ms=capsize,
                color=bar_color,
            )
            caplines.append(cap)

    if yerr is not None:
        e = np.asarray(yerr, dtype=float)
        lo, hi = y - (e[0] if e.ndim == 2 else e), y + (e[1] if e.ndim == 2 else e)
        segs = _wrap_to_segments(np.repeat(x, 3), bar_extent(lo, hi), wrapx, wrapy)
        bars = LineCollection(segs, colors=bar_color, lw=bar_lw)
        ax.add_collection(bars)
        barlinecols.append(bars)
        add_caps(x, lo, "_")
        add_caps(x, hi, "_")
    if xerr is not None:
        e = np.asarray(xerr, dtype=float)
        lo, hi = x - (e[0] if e.ndim == 2 else e), x + (e[1] if e.ndim == 2 else e)
        segs = _wrap_to_segments(bar_extent(lo, hi), np.repeat(y, 3), wrapx, wrapy)
        bars = LineCollection(segs, colors=bar_color, lw=bar_lw)
        ax.add_collection(bars)
        barlinecols.append(bars)
        add_caps(lo, y, "|")
        add_caps(hi, y, "|")

    container = ErrorbarContainer(
        # data_line may be None with fmt="none", as in ax.errorbar itself
        (data_line, tuple(caplines), tuple(barlinecols)),  # type: ignore[arg-type]
        has_xerr=xerr is not None,
        has_yerr=yerr is not None,
        label=label,
    )
    ax.add_container(container)
    return container
