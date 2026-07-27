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
    _band_vertices,
    _error_bounds,
    _nan_joined_extents,
    _stairs_polyline,
    _step_polyline,
    _wrap_to_segments,
    wrap_line,
    wrap_points,
)

__all__ = [
    "set_wrap",
    "plot_wrapped",
    "scatter_wrapped",
    "fill_between_wrapped",
    "step_wrapped",
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

# Valid step placements, as in ax.step(where=...) and ax.fill_between(step=...).
_STEP_WHERE = ("pre", "post", "mid")


def _check_step_where(func: str, name: str, value: str) -> None:
    """Validate a step placement, naming the mpl_wrap function and its argument."""
    if value not in _STEP_WHERE:
        raise ValueError(f"{func}() {name}={value!r} is not one of {list(_STEP_WHERE)}.")


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


def _add_band_patch(
    ax: Axes,
    verts: np.ndarray,
    codes: np.ndarray,
    wrapx: np.ndarray | None,
    wrapy: np.ndarray | None,
    kwargs: dict[str, Any],
) -> PathPatch:
    """Add a tiled band path to the axes as a clipped patch."""
    kwargs.setdefault("linewidth", 0)
    patch = PathPatch(Path(verts, codes), **kwargs)
    patch.set_transform(ax.transData)
    # Clipped to the window, so keep its huge path out of datalim (add_artist) and
    # layout (set_in_layout) - both would otherwise walk every tiled vertex.
    ax.add_artist(patch)
    _clip_patch_to_window(ax, patch, wrapx, wrapy)
    patch.set_in_layout(False)
    return patch


def fill_between_wrapped(
    ax: Axes,
    x: Any,
    y1: Any,
    y2: Any = 0,
    where: Any = None,
    interpolate: bool = False,
    step: str | None = None,
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
    y1, y2 : array-like or float
        The band edges (continuous / unwrapped), in either order. ``y2``
        defaults to 0.
    where : array-like of bool, optional
        Fill only where True, as in ``ax.fill_between``. The band is drawn once
        per contiguous run; non-finite samples break runs too.
    interpolate : bool, default False
        Extend each run to the interpolated point where ``y1`` and ``y2``
        actually cross, as in ``ax.fill_between``.
    step : {"pre", "post", "mid"}, optional
        Step the band's edges before wrapping, as in ``ax.fill_between``. None
        (the default) interpolates linearly between samples.
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
        The band artist, added to the axes (excluded from data limits and
        layout). Note that this is a ``PathPatch`` in ``ax.patches``, where
        ``ax.fill_between`` returns a collection in ``ax.collections``.
    """
    if step is not None:
        _check_step_where("fill_between_wrapped", "step", step)
    x = _to_num(ax.xaxis, x)
    y1 = _to_num(ax.yaxis, y1)
    y2 = _to_num(ax.yaxis, y2)
    wx = _resolve_wrap(ax, "x", wrapx)
    wy = _resolve_wrap(ax, "y", wrapy)
    verts, codes = _band_vertices(
        x, y1, y2, where=where, interpolate=interpolate, step=step, wrapx=wx, wrapy=wy
    )
    return _add_band_patch(ax, verts, codes, wx, wy, kwargs)


def step_wrapped(
    ax: Axes,
    x: Any,
    y: Any,
    *args: Any,
    where: str = "pre",
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> list[Line2D]:
    """Draw a continuous (unwrapped) step series on a wrapped axis.

    Mirrors ``ax.step`` - n x-values, n y-values, and a ``where`` policy - with
    optional ``wrapx`` and/or ``wrapy`` (min, max) windows. The steps are built
    as a tread/riser polyline and wrapped, so seam-crossing risers route to the
    window edges. Rendered with ``ax.plot`` on the expanded polyline rather than
    via ``drawstyle``. Datetime data and windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x, y : array-like
        Continuous (unwrapped) data coordinates, of equal length.
    where : {"pre", "post", "mid"}, default "pre"
        Where the risers fall relative to the x positions, as in ``ax.step``.
    *args, **kwargs
        Forwarded to ``ax.plot`` (format string, styling, ...).
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The plotted line artists, as from ``ax.step``.
    """
    _check_step_where("step_wrapped", "where", where)
    x, y, wx, wy = _prepare_xy(ax, x, y, wrapx, wrapy)
    step_x, step_y = _step_polyline(x, y, where=where)
    return ax.plot(*wrap_line(step_x, step_y, wrapx=wx, wrapy=wy), *args, **kwargs)


def stairs_wrapped(
    ax: Axes,
    values: Any,
    edges: Any = None,
    *,
    orientation: str = "vertical",
    baseline: Any = 0,
    fill: bool = False,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> Union[list[Line2D], PathPatch]:
    """Draw a continuous (unwrapped) staircase on a wrapped axis.

    Mirrors ``ax.stairs`` with optional ``wrapx`` and/or ``wrapy`` (min, max)
    windows. The staircase is turned into a tread/riser polyline and wrapped, so
    seam-crossing risers route to the edges. As in ``ax.stairs``, the default
    ``baseline=0`` drops both ends down to the baseline (pass ``baseline=None``
    for the bare staircase). For the ``ax.step`` signature (n x-values against n
    y-values) use `step_wrapped`. Datetime data and windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    values : array-like
        Step heights (continuous / unwrapped).
    edges : array-like, optional
        Bin edges, one longer than ``values``. Defaults to ``arange(len(values) + 1)``.
    orientation : {"vertical", "horizontal"}, default "vertical"
        Whether ``values`` run along y (edges along x) or the reverse.
    baseline : float, array-like or None, default 0
        The value the ends drop to, or the lower edge of the fill. None leaves
        the staircase open, and cannot be combined with ``fill``.
    fill : bool, default False
        Fill between the staircase and the baseline instead of drawing a line.
        The fill is tiled and clipped exactly as in `fill_between_wrapped`.
    **kwargs
        Forwarded to ``ax.plot`` (styling, ...), or to the
        ``matplotlib.patches.PathPatch`` when ``fill=True``.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    list of matplotlib.lines.Line2D, or matplotlib.patches.PathPatch if ``fill``
        The plotted artists. Note that these are the ``ax.plot`` /
        `fill_between_wrapped` artists, where ``ax.stairs`` returns a single
        ``StepPatch``.
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"stairs_wrapped() orientation={orientation!r} is not one of "
            f"['vertical', 'horizontal']."
        )
    # Vertical puts values on y and edges on x; horizontal swaps the two axes.
    vertical = orientation == "vertical"
    value_axis, edge_axis = (ax.yaxis, ax.xaxis) if vertical else (ax.xaxis, ax.yaxis)
    values = _to_num(value_axis, values)
    edges = np.arange(len(values) + 1, dtype=float) if edges is None else _to_num(edge_axis, edges)
    base: np.ndarray | None = None if baseline is None else _to_num(value_axis, baseline)
    wx = _resolve_wrap(ax, "x", wrapx)
    wy = _resolve_wrap(ax, "y", wrapy)
    # Build in "vertical" space (edges along x), then swap back if horizontal.
    w_edge, w_value = (wx, wy) if vertical else (wy, wx)

    if not fill:
        step_e, step_v = _stairs_polyline(values, edges, base)
        line_x, line_y = (step_e, step_v) if vertical else (step_v, step_e)
        return ax.plot(*wrap_line(line_x, line_y, wrapx=wx, wrapy=wy), **kwargs)

    # Filled: a band between the staircase and the (scalar or stepped) baseline.
    if base is None:
        raise ValueError(
            "stairs_wrapped() cannot fill with baseline=None: pass a baseline "
            "value or array to fill against."
        )
    band_e, band_v = _stairs_polyline(values, edges)
    band_base = base if base.ndim == 0 else np.repeat(base, 2)
    verts, codes = _band_vertices(band_e, band_v, band_base, wrapx=w_edge, wrapy=w_value)
    if not vertical:
        verts = verts[:, ::-1]
    return _add_band_patch(ax, verts, codes, wx, wy, kwargs)


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
