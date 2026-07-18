"""Helper functions for plotting wrapped, angular, or periodic data on matplotlib axes.

Free functions that mirror the core matplotlib plotting methods, taking an
``Axes`` plus ``wrapx`` / ``wrapy`` (min, max) windows: continuous (unwrapped)
data is folded into the window, with lines routed to the window edges at each
seam crossing instead of drawing jump artifacts. ``set_wrap`` stores a window
on an axes so subsequent calls pick it up automatically.
"""

from collections.abc import Iterable
from typing import Any, Union, overload

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from mpl_wrap.data import (
    _error_bounds,
    _nan_joined_extents,
    _saturated_band_vertices,
    _tiled_band_vertices,
    _wrap_to_segments,
    wrap_line,
    wrap_points,
)

__all__ = [
    "set_wrap",
    "plot_wrapped",
    "scatter_wrapped",
    "fill_between_wrapped",
    "stairs_wrapped",
    "errorbar_wrapped",
]

# Wrap window spec:
# - (min, max) pair in data units (datetimes allowed)
# - True to require the window stored by set_wrap
# - False to explicitly disable wrapping on an axis with a stored window
# - None to fall back to the window stored by set_wrap (if any).
WrapSpec = Union[Iterable[Any], bool, None]

_WINDOW_ATTR = "_mpl_wrap_windows"


# typing overloads
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

    True requires a stored window, and False explicitly disables wrapping even
    when a stored window exists.
    """
    if wrap is False:
        return None
    stored: dict[str, np.ndarray] = getattr(ax, _WINDOW_ATTR, {})
    if wrap is True:
        if name not in stored:
            raise ValueError(
                f"wrap{name}=True, but no {name} wrap window is stored on this axes. "
                f"Call set_wrap(ax, wrap{name}=...) first."
            )
        return stored[name]
    if wrap is not None:
        axis = ax.xaxis if name == "x" else ax.yaxis
        return _to_num(axis, wrap)
    return stored.get(name)


def _prepare_xy(
    ax: Axes, x: Any, y: Any, wrapx: WrapSpec, wrapy: WrapSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Convert x/y data to numeric arrays and resolve their wrap windows."""
    x_num = _to_num(ax.xaxis, x)
    y_num = _to_num(ax.yaxis, y)
    wx = _resolve_wrap(ax, "x", wrapx)
    wy = _resolve_wrap(ax, "y", wrapy)
    return x_num, y_num, wx, wy


def set_wrap(
    ax: Axes,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    *,
    set_lims: bool = True,
    seam_lines: bool = False,
    seam_kwargs: dict[str, Any] | None = None,
) -> Axes:
    """Store wrap window(s) on an axes so the plotting helpers use them by default.

    After ``set_wrap(ax, wrapy=(0, 360))``, helpers called on ``ax`` without an
    explicit ``wrapy`` wrap into the stored window. An explicit per-call window
    still overrides, and ``wrapx=False`` / ``wrapy=False`` disables wrapping for
    a single call. Calling ``set_wrap`` again updates only the windows given
    (pass ``False`` to clear a stored window).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window for the x/y axis, in data units (datetimes allowed).
        ``False`` clears a previously stored window, and None leaves it unchanged.
    set_lims : bool, default True
        Set the axis limits of each given window to the window.
    seam_lines : bool, default False
        Draw lines at the window edges of each given window.
    seam_kwargs : dict, optional
        Overrides for the seam line style (default ``color="k", linewidth=0.8``).

    Returns
    -------
    matplotlib.axes.Axes
        The same axes, for chaining.
    """
    windows: dict[str, np.ndarray] = dict(getattr(ax, _WINDOW_ATTR, {}))
    style: dict[str, Any] = {"color": "k", "linewidth": 0.8}
    style.update(seam_kwargs or {})
    for name, wrap, axis, set_lim, seam in (
        ("x", wrapx, ax.xaxis, ax.set_xlim, ax.axvline),
        ("y", wrapy, ax.yaxis, ax.set_ylim, ax.axhline),
    ):
        if wrap is None:
            continue
        if wrap is True:
            raise ValueError(f"wrap{name}=True is not valid in set_wrap. Pass a (min, max) window.")
        if wrap is False:
            windows.pop(name, None)
            continue
        w = _to_num(axis, wrap)
        windows[name] = w
        if set_lims:
            set_lim(w[0], w[1])
        if seam_lines:
            seam(w[0], **style)
            seam(w[1], **style)
    setattr(ax, _WINDOW_ATTR, windows)
    return ax


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
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The plotted line artists, as from ``ax.plot``.
    """
    x, y, wx, wy = _prepare_xy(ax, x, y, wrapx, wrapy)
    xs, ys = wrap_line(x, y, wrapx=wx, wrapy=wy)
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
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter artist, as from ``ax.scatter``.
    """
    x, y, wx, wy = _prepare_xy(ax, x, y, wrapx, wrapy)
    return ax.scatter(*wrap_points(x, y, wrapx=wx, wrapy=wy), *args, **kwargs)


def _clip_patch_to_window(
    ax: Axes,
    patch: PathPatch,
    wrapx: np.ndarray | None,
    wrapy: np.ndarray | None,
) -> None:
    """Clip a filled patch to the wrap window(s), a wrapped axis in data, the other full."""
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
        The band edges (continuous / unwrapped), in either order.
    **kwargs
        Forwarded to the ``matplotlib.patches.PathPatch`` (color, alpha, ...).
        ``linewidth`` defaults to 0.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

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

    # y-only wrap: saturated x-runs collapse to one rectangle each, else tile in x/y.
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
        Bin edges, one longer than ``values``. Defaults to ``arange(len(values) + 1)``.
    **kwargs
        Forwarded to ``ax.plot`` (``baseline`` and ``fill`` are ignored).
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

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
    return ax.plot(*wrap_line(step_x, step_y, wrapx=wx, wrapy=wy), **kwargs)


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
        Format string for the data markers. ``"none"`` suppresses them.
    ecolor, elinewidth, capsize
        Bar and cap styling, as in ``ax.errorbar``.
    **kwargs
        Forwarded to ``ax.plot`` for the data line/markers.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    matplotlib.container.ErrorbarContainer
        Container of (data line, caplines, barlinecols), as from ``ax.errorbar``.
    """
    x, y, wrapx, wrapy = _prepare_xy(ax, x, y, wrapx, wrapy)
    label = kwargs.pop("label", None)

    # Data line / markers at the wrapped centres ('none' suppresses them, as in
    # ax.errorbar). Only the container carries the legend label, so the entry
    # shows once, with the bar-and-marker handle.
    data_line: Line2D | None = None
    if fmt != "none":
        drawn = ax.plot(*wrap_points(x, y, wrapx=wrapx, wrapy=wrapy), fmt, **kwargs)
        data_line = drawn[0] if drawn else None

    bar_color = (
        ecolor if ecolor is not None else (data_line.get_color() if data_line is not None else "C0")
    )
    bar_lw = elinewidth if elinewidth is not None else mpl.rcParams["lines.linewidth"]
    barlinecols: list[LineCollection] = []
    caplines: list[Line2D] = []

    def add_caps(cx: np.ndarray, cy: np.ndarray, marker: str) -> None:
        if capsize:
            (cap,) = ax.plot(
                *wrap_points(cx, cy, wrapx=wrapx, wrapy=wrapy),
                linestyle="none",
                marker=marker,
                ms=capsize,
                color=bar_color,
            )
            caplines.append(cap)

    for error, positions, values, horizontal, cap_marker in (
        (yerr, x, y, False, "_"),
        (xerr, y, x, True, "|"),
    ):
        if error is None:
            continue
        lo, hi = _error_bounds(values, error)
        fixed = np.repeat(positions, 3)
        extents = _nan_joined_extents(lo, hi)
        bar_x, bar_y = (extents, fixed) if horizontal else (fixed, extents)
        segs = _wrap_to_segments(bar_x, bar_y, wrapx, wrapy)
        bars = LineCollection(segs, colors=bar_color, lw=bar_lw)
        ax.add_collection(bars)
        barlinecols.append(bars)
        cap_lo_x, cap_lo_y = (lo, positions) if horizontal else (positions, lo)
        cap_hi_x, cap_hi_y = (hi, positions) if horizontal else (positions, hi)
        add_caps(cap_lo_x, cap_lo_y, cap_marker)
        add_caps(cap_hi_x, cap_hi_y, cap_marker)

    container = ErrorbarContainer(
        # data_line may be None with fmt="none", as in ax.errorbar itself
        (data_line, tuple(caplines), tuple(barlinecols)),  # type: ignore[arg-type]
        has_xerr=xerr is not None,
        has_yerr=yerr is not None,
        label=label,
    )
    ax.add_container(container)
    return container
