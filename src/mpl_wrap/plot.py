"""Helper functions for plotting wrapped, angular, or periodic data on matplotlib axes.

Free functions that mirror the core matplotlib plotting methods, taking an
``Axes`` plus ``wrapx`` / ``wrapy`` (min, max) windows: continuous (unwrapped)
data is folded into the window, with lines routed to the window edges at each
seam crossing instead of drawing jump artifacts. ``set_wrap`` stores a window
on an axes so subsequent calls pick it up automatically.
"""

from collections.abc import Iterable
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path

from mpl_wrap.artists import WrapFillBetween, WrapStepPatch
from mpl_wrap.data import (
    _band_extent,
    _band_vertices,
    _error_bounds,
    _nan_joined_extents,
    _polyline_path,
    _stairs_polyline,
    _step_polyline_samples,
    _wrap_line_samples,
    _wrap_span,
    _wrap_to_segments,
    wrap_line,
    wrap_points,
)

__all__ = [
    "axhspan_wrapped",
    "axvspan_wrapped",
    "errorbar_wrapped",
    "fill_between_wrapped",
    "fill_betweenx_wrapped",
    "hlines_wrapped",
    "plot_wrapped",
    "scatter_wrapped",
    "set_wrap",
    "stairs_wrapped",
    "step_wrapped",
    "vlines_wrapped",
]

# Wrap window spec:
# - (min, max) pair in data units (datetimes allowed)
# - True to require the window stored by set_wrap
# - False to explicitly disable wrapping on an axis with a stored window
# - None to fall back to the window stored by set_wrap (if any).
WrapSpec = Iterable[Any] | bool | None

_WINDOW_ATTR = "_mpl_wrap_windows"

# Valid step placements, as in ax.step(where=...) and ax.fill_between(step=...).
_STEP_WHERE = ("pre", "post", "mid")


def _window_clip(ax: Axes, artist: Any, wrapx: np.ndarray | None, wrapy: np.ndarray | None) -> None:
    """Clip an artist to the wrap window(s), a wrapped axis in data, the other full.

    Clipping with a Path rather than a Rectangle patch leaves the artist's clip
    box in place, so a window wider than the view cannot spill outside the axes.
    """
    if wrapx is not None and wrapy is not None:
        (x0, x1), (y0, y1), transform = wrapx, wrapy, ax.transData
    elif wrapy is not None:
        (x0, x1), (y0, y1), transform = (0.0, 1.0), wrapy, ax.get_yaxis_transform()
    elif wrapx is not None:
        (x0, x1), (y0, y1), transform = wrapx, (0.0, 1.0), ax.get_xaxis_transform()
    else:
        return
    ring = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=float)
    artist.set_clip_path(Path(ring), transform)


def _plot_marked(
    ax: Axes, xs: np.ndarray, ys: np.ndarray, samples: np.ndarray, args: Any, kwargs: Any
) -> list[Line2D]:
    """Plot a wrapped polyline, keeping markers on the data points.

    Seam routing and step expansion insert vertices that are not data points.
    markevery pins the markers to the samples, as matplotlib does for drawstyles.
    """
    if len(samples) != len(xs):
        kwargs.setdefault("markevery", samples.tolist())
    return ax.plot(xs, ys, *args, **kwargs)


def _check_step_where(func: str, name: str, value: str) -> None:
    """Validate a step placement, naming the mpl_wrap function and its argument."""
    if value not in _STEP_WHERE:
        raise ValueError(f"{func}() {name}={value!r} is not one of {list(_STEP_WHERE)}.")


def _to_num(axis: Axis, values: Any) -> np.ndarray:
    """Register units on the axis and return values in matplotlib's numeric form.

    Lets datetime (and other unit-ful) inputs be wrapped: the axis learns the
    converter, so ticks still format correctly, and we get plain floats to do the
    wrapping arithmetic on. Already-numeric data passes straight through.
    """
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


def _windows(
    ax: Axes, wrapx: WrapSpec, wrapy: WrapSpec
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve both wrap specs against the axes."""
    return _resolve_wrap(ax, "x", wrapx), _resolve_wrap(ax, "y", wrapy)


def _prepare_xy(
    ax: Axes, x: Any, y: Any, wrapx: WrapSpec, wrapy: WrapSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Convert x/y data to numeric arrays and resolve their wrap windows."""
    return (_to_num(ax.xaxis, x), _to_num(ax.yaxis, y), *_windows(ax, wrapx, wrapy))


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
    xs, ys, samples = _wrap_line_samples(x, y, wrapx=wx, wrapy=wy)
    return _plot_marked(ax, xs, ys, samples, args, kwargs)


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


def hlines_wrapped(
    ax: Axes,
    y: Any,
    xmin: Any,
    xmax: Any,
    colors: Any = None,
    linestyles: Any = "solid",
    *,
    label: str = "",
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> LineCollection:
    """Draw horizontal lines on a wrapped axis, splitting each span at the seam.

    Mirrors ``ax.hlines`` with optional ``wrapx`` and/or ``wrapy`` (min, max)
    windows, and returns the same artist. A span crossing an x seam is split so
    it shows at both edges, and one at least an x-period long sweeps the window.
    ``y`` is folded into the y window. Datetime data and windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    y : float or array-like
        The heights of the lines.
    xmin, xmax : float or array-like
        The span of each line (continuous / unwrapped).
    colors, linestyles, label
        As in ``ax.hlines``. A list given per line is expanded with it, so each
        piece of a split line keeps that line's style.
    **kwargs
        Forwarded to the ``LineCollection``, as in ``ax.hlines``.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    matplotlib.collections.LineCollection
        The lines, as from ``ax.hlines``.
    """
    return _wrapped_lines(ax, True, y, xmin, xmax, colors, linestyles, label, wrapx, wrapy, kwargs)


def vlines_wrapped(
    ax: Axes,
    x: Any,
    ymin: Any,
    ymax: Any,
    colors: Any = None,
    linestyles: Any = "solid",
    *,
    label: str = "",
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> LineCollection:
    """Draw vertical lines on a wrapped axis, splitting each span at the seam.

    The `hlines_wrapped` mirror of ``ax.vlines``: a span crossing a y seam is
    split so it shows at both edges, one at least a y-period long sweeps the
    window, and ``x`` is folded into the x window. Datetime data and windows are
    accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x : float or array-like
        The positions of the lines.
    ymin, ymax : float or array-like
        The span of each line (continuous / unwrapped).
    colors, linestyles, label
        As in ``ax.vlines``. A list given per line is expanded with it, so each
        piece of a split line keeps that line's style.
    **kwargs
        Forwarded to the ``LineCollection``, as in ``ax.vlines``.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    matplotlib.collections.LineCollection
        The lines, as from ``ax.vlines``.
    """
    return _wrapped_lines(ax, False, x, ymin, ymax, colors, linestyles, label, wrapx, wrapy, kwargs)


def axhspan_wrapped(
    ax: Axes,
    ymin: Any,
    ymax: Any,
    xmin: float = 0,
    xmax: float = 1,
    *,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> list[Rectangle]:
    """Shade a horizontal band on a wrapped axis, folding it into the window.

    Mirrors ``ax.axhspan`` with an optional ``wrapy`` (min, max) window. The
    band is folded into the window: one rectangle if it fits, two if it
    straddles the seam, and the whole window if it spans a period or more.
    ``xmin`` and ``xmax`` are in axes coordinates, as in ``ax.axhspan``, so only
    the y window applies. Datetime data and windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    ymin, ymax : float
        The band's edges, in data coordinates (continuous / unwrapped).
    xmin, xmax : float, default 0 and 1
        The band's horizontal extent, in axes coordinates.
    **kwargs
        Forwarded to the ``Rectangle``, as in ``ax.axhspan``. Only the first
        piece of a split band takes a ``label``, so the legend shows one entry.
    wrapy : (min, max) or False, optional
        Wrap window, defaulting to the window stored by `set_wrap`. ``True``
        requires the stored window, and ``False`` disables wrapping.

    Returns
    -------
    list of matplotlib.patches.Rectangle
        The pieces of the band. A list, where ``ax.axhspan`` returns a single
        ``Rectangle``, because a band across the seam really is two rectangles.
    """
    return _wrapped_span(ax, False, ymin, ymax, xmin, xmax, wrapy, kwargs)


def axvspan_wrapped(
    ax: Axes,
    xmin: Any,
    xmax: Any,
    ymin: float = 0,
    ymax: float = 1,
    *,
    wrapx: WrapSpec = None,
    **kwargs: Any,
) -> list[Rectangle]:
    """Shade a vertical band on a wrapped axis, folding it into the window.

    The `axhspan_wrapped` mirror of ``ax.axvspan``: the band spans ``xmin`` to
    ``xmax`` in data coordinates, folded into the ``wrapx`` window, and ``ymin``
    and ``ymax`` are in axes coordinates. Datetime data and windows are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    xmin, xmax : float
        The band's edges, in data coordinates (continuous / unwrapped).
    ymin, ymax : float, default 0 and 1
        The band's vertical extent, in axes coordinates.
    **kwargs
        Forwarded to the ``Rectangle``, as in ``ax.axvspan``. Only the first
        piece of a split band takes a ``label``.
    wrapx : (min, max) or False, optional
        Wrap window, defaulting to the window stored by `set_wrap`. ``True``
        requires the stored window, and ``False`` disables wrapping.

    Returns
    -------
    list of matplotlib.patches.Rectangle
        The pieces of the band, where ``ax.axvspan`` returns a single ``Rectangle``.
    """
    return _wrapped_span(ax, True, xmin, xmax, ymin, ymax, wrapx, kwargs)


def _wrapped_span(
    ax: Axes,
    vertical: bool,
    lo: Any,
    hi: Any,
    start: float,
    end: float,
    wrap: WrapSpec,
    kwargs: dict[str, Any],
) -> list[Rectangle]:
    """Fold a span into the window and draw its piece(s) with ax.axhspan / ax.axvspan."""
    name = "x" if vertical else "y"
    axis = ax.xaxis if vertical else ax.yaxis
    bounds = _to_num(axis, [lo, hi])
    draw = ax.axvspan if vertical else ax.axhspan
    label = kwargs.pop("label", None)
    pieces = _wrap_span(float(bounds[0]), float(bounds[1]), _resolve_wrap(ax, name, wrap))
    # Only the first piece is labelled, so a split band shows one legend entry.
    labels = [label if label is not None else "_nolegend_", *["_nolegend_"] * (len(pieces) - 1)]
    return [draw(a, b, start, end, label=lab, **kwargs) for (a, b), lab in zip(pieces, labels)]


def _per_piece(style: Any, counts: np.ndarray) -> Any:
    """Repeat a per-line style over the pieces each line was split into.

    A list is one entry per line. Anything else (a colour, a linestyle string, a
    dash tuple) applies to every line, as in ``ax.hlines``.
    """
    if not isinstance(style, list) or len(style) != len(counts):
        return style
    return [value for value, count in zip(style, counts) for _ in range(count)]


def _wrapped_lines(
    ax: Axes,
    horizontal: bool,
    pos: Any,
    lo: Any,
    hi: Any,
    colors: Any,
    linestyles: Any,
    label: str,
    wrapx: WrapSpec,
    wrapy: WrapSpec,
    kwargs: dict[str, Any],
) -> LineCollection:
    """Split wrapped spans into pieces and hand them back to ax.hlines / ax.vlines."""
    pos_axis, span_axis = (ax.yaxis, ax.xaxis) if horizontal else (ax.xaxis, ax.yaxis)
    pos = _to_num(pos_axis, pos)
    lo = _to_num(span_axis, lo)
    hi = _to_num(span_axis, hi)
    wx, wy = _windows(ax, wrapx, wrapy)
    pos, lo, hi = np.broadcast_arrays(np.atleast_1d(pos), np.atleast_1d(lo), np.atleast_1d(hi))

    positions: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    counts = np.zeros(len(pos), dtype=int)
    for i, (p, a, b) in enumerate(zip(pos, lo, hi)):
        span, fixed = np.array([a, b], dtype=float), np.array([p, p], dtype=float)
        seg_x, seg_y = (span, fixed) if horizontal else (fixed, span)
        for piece in _wrap_to_segments(seg_x, seg_y, wx, wy):
            along, across = (piece[:, 0], piece[:, 1]) if horizontal else (piece[:, 1], piece[:, 0])
            positions.append(across[0])
            starts.append(along.min())
            ends.append(along.max())
            counts[i] += 1

    draw = ax.hlines if horizontal else ax.vlines
    return draw(
        positions,
        starts,
        ends,
        colors=_per_piece(colors, counts),
        linestyles=_per_piece(linestyles, counts),
        label=label,
        **kwargs,
    )


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
) -> WrapFillBetween:
    """Fill between two continuous (unwrapped) series on a wrapped axis.

    Mirrors ``ax.fill_between`` with optional ``wrapx`` and/or ``wrapy``
    (min, max) windows, and returns the same artist type. The band is tiled at
    every period offset (in x and/or y) into one compound path clipped to the
    window, so the union fills once with no double alpha, and a band at least a
    y-period wide fills the whole window as "fully uncertain". The fill is the
    one helper that clips rather than routing to edges. Datetime data and
    windows are accepted.

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
        per contiguous run, and non-finite samples break runs too.
    interpolate : bool, default False
        Extend each run to the interpolated point where ``y1`` and ``y2``
        actually cross, as in ``ax.fill_between``.
    step : {"pre", "post", "mid"}, optional
        Step the band's edges before wrapping, as in ``ax.fill_between``. None
        (the default) interpolates linearly between samples.
    **kwargs
        Forwarded to the ``FillBetweenPolyCollection`` (color, alpha, ...), as
        in ``ax.fill_between``. An edge, if drawn, follows the wrapped band -
        once per period it sweeps.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    mpl_wrap.artists.WrapFillBetween
        The band artist, a ``FillBetweenPolyCollection`` in ``ax.collections``
        as from ``ax.fill_between``. Its path is clipped to the window, so its
        vertices and data limits cover only what is drawn.
    """
    return _fill_band(
        ax, "x", x, y1, y2, where, interpolate, step, wrapx, wrapy, "fill_between_wrapped", kwargs
    )


def fill_betweenx_wrapped(
    ax: Axes,
    y: Any,
    x1: Any,
    x2: Any = 0,
    where: Any = None,
    interpolate: bool = False,
    step: str | None = None,
    *,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    **kwargs: Any,
) -> WrapFillBetween:
    """Fill between two continuous (unwrapped) series on a wrapped axis, along y.

    The `fill_between_wrapped` mirror of ``ax.fill_betweenx``: the band runs
    along y, bounded by ``x1`` and ``x2``, and is wrapped into the ``wrapx``
    and/or ``wrapy`` (min, max) windows the same way. Datetime data and windows
    are accepted.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    y : array-like
        Continuous (unwrapped) y coordinates.
    x1, x2 : array-like or float
        The band edges (continuous / unwrapped), in either order. ``x2``
        defaults to 0.
    where, interpolate, step
        As in `fill_between_wrapped`, along y.
    **kwargs
        Forwarded to the ``FillBetweenPolyCollection``, as in ``ax.fill_betweenx``.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    mpl_wrap.artists.WrapFillBetween
        The band artist, a ``FillBetweenPolyCollection`` in ``ax.collections``
        as from ``ax.fill_betweenx``.
    """
    return _fill_band(
        ax, "y", y, x1, x2, where, interpolate, step, wrapx, wrapy, "fill_betweenx_wrapped", kwargs
    )


def _fill_band(
    ax: Axes,
    t_direction: Literal["x", "y"],
    t: Any,
    f1: Any,
    f2: Any,
    where: Any,
    interpolate: bool,
    step: str | None,
    wrapx: WrapSpec,
    wrapy: WrapSpec,
    func: str,
    kwargs: dict[str, Any],
) -> WrapFillBetween:
    """Add a wrapped band running along ``t_direction`` ("x" or "y")."""
    if step is not None:
        _check_step_where(func, "step", step)
    t_axis, f_axis = (ax.xaxis, ax.yaxis) if t_direction == "x" else (ax.yaxis, ax.xaxis)
    t = _to_num(t_axis, t)
    f1 = _to_num(f_axis, f1)
    f2 = _to_num(f_axis, f2)
    wx, wy = _windows(ax, wrapx, wrapy)
    # Take the next fill colour when none is given, as ax.fill_between does.
    if not mpl.rcParams["_internal.classic_mode"]:  # type: ignore[index]
        kwargs = mpl.cbook.normalize_kwargs(kwargs, Collection)
        if not any(c in kwargs for c in ("color", "facecolor")):
            kwargs["facecolor"] = ax._get_patches_for_fill.get_next_color()  # type: ignore[attr-defined]
    band = WrapFillBetween(
        t_direction,
        t,
        f1,
        f2,
        where=where,
        interpolate=interpolate,
        step=step,
        wrapx=wx,
        wrapy=wy,
        **kwargs,
    )
    ax.add_collection(band)
    _window_clip(ax, band, wx, wy)
    return band


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
    step_x, step_y, steps = _step_polyline_samples(x, y, where)
    xs, ys, pos = _wrap_line_samples(step_x, step_y, wrapx=wx, wrapy=wy)
    return _plot_marked(ax, xs, ys, pos[steps], args, kwargs)


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
) -> WrapStepPatch:
    """Draw a continuous (unwrapped) staircase on a wrapped axis.

    Mirrors ``ax.stairs`` with optional ``wrapx`` and/or ``wrapy`` (min, max)
    windows, and returns the same artist type. The staircase is turned into a
    tread/riser polyline and wrapped, so seam-crossing risers route to the
    edges. As in ``ax.stairs``, the default ``baseline=0`` drops both ends down
    to the baseline (pass ``baseline=None`` for the bare staircase). For the
    ``ax.step`` signature (n x-values against n y-values) use `step_wrapped`.
    Datetime data and windows are accepted.

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
        Forwarded to the ``matplotlib.patches.StepPatch`` (edgecolor,
        facecolor, alpha, ...), as in ``ax.stairs``.
    wrapx, wrapy : (min, max) or False, optional
        Wrap window per axis, defaulting to the window stored by `set_wrap`.
        ``True`` requires the stored window, and ``False`` disables wrapping
        for this call.

    Returns
    -------
    mpl_wrap.artists.WrapStepPatch
        The staircase artist, a ``StepPatch`` in ``ax.patches`` as from
        ``ax.stairs``. ``get_data`` returns the values and edges as passed.
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError(
            f"stairs_wrapped() orientation={orientation!r} is not one of "
            f"['vertical', 'horizontal']."
        )
    if fill and baseline is None:
        raise ValueError(
            "stairs_wrapped() cannot fill with baseline=None: pass a baseline "
            "value or array to fill against."
        )
    # Vertical puts values on y and edges on x, horizontal swaps them.
    vertical = orientation == "vertical"
    value_axis, edge_axis = (ax.yaxis, ax.xaxis) if vertical else (ax.xaxis, ax.yaxis)
    values = _to_num(value_axis, values)
    edges = np.arange(len(values) + 1, dtype=float) if edges is None else _to_num(edge_axis, edges)
    base = None if baseline is None else _to_num(value_axis, baseline)
    wx, wy = _windows(ax, wrapx, wrapy)

    # Colour defaults exactly as ax.stairs picks them.
    color = kwargs.pop("color", None)
    if color is None:
        color = ax._get_lines.get_next_color()  # type: ignore[attr-defined]
    if fill:
        kwargs.setdefault("linewidth", 0)
        kwargs.setdefault("facecolor", color)
    else:
        kwargs.setdefault("edgecolor", color)

    def build(
        values: np.ndarray, edges: np.ndarray, base: np.ndarray | None, orientation: str
    ) -> Path:
        """Wrap the staircase, built in "vertical" space and swapped if horizontal."""
        vertical = orientation == "vertical"
        if not fill:
            step_e, step_v = _stairs_polyline(values, edges, base)
            line_x, line_y = (step_e, step_v) if vertical else (step_v, step_e)
            return _polyline_path(*wrap_line(line_x, line_y, wrapx=wx, wrapy=wy))
        # Filled: a band between the staircase and the (scalar or stepped) baseline.
        assert base is not None  # fill=True without a baseline is rejected above
        band_e, band_v = _stairs_polyline(values, edges)
        band_base = base if base.ndim == 0 else np.repeat(base, 2)
        w_edge, w_value = (wx, wy) if vertical else (wy, wx)
        verts, codes = _band_vertices(band_e, band_v, band_base, wrapx=w_edge, wrapy=w_value)
        if not vertical:
            verts = verts[:, ::-1]
        return Path(verts, codes)

    patch = WrapStepPatch(
        values, edges, baseline=base, orientation=orientation, fill=fill, build=build, **kwargs
    )
    if fill:
        # add_patch would walk every tiled vertex, so add it and set the limits here.
        ax.add_artist(patch)
        _window_clip(ax, patch, wx, wy)
        verts = np.asarray(patch.get_path().vertices, dtype=float)
        ax.update_datalim(_band_extent(verts, wx, wy).get_points())
    else:
        ax.add_patch(patch)
    # The baseline anchors autoscaling as in ax.stairs, but only on an unwrapped
    # axis, where it is drawn at its own value.
    if base is not None and (wy if vertical else wx) is None:
        low = float(np.min(base))
        (patch.sticky_edges.y if vertical else patch.sticky_edges.x).append(low)
        ax.update_datalim([(edges[0], low) if vertical else (low, edges[0])])
    ax._request_autoscale_view()  # type: ignore[attr-defined]
    return patch


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
        Format string for the data line and markers. ``"none"`` suppresses them.
    ecolor, elinewidth, capsize
        Bar and cap styling, as in ``ax.errorbar``. ``elinewidth`` and
        ``capsize`` fall back to their rcParams, and ``capsize`` is the cap's
        half-width, as in ``ax.errorbar``.
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
    kwargs.setdefault("zorder", 2)

    # The data line follows the wrapped polyline with markers on the data points,
    # and 'none' suppresses it, as in ax.errorbar. Only the container carries the
    # label, so the legend shows one bar-and-marker entry.
    data_line: Line2D | None = None
    if fmt.lower() != "none":
        xs, ys, samples = _wrap_line_samples(x, y, wrapx=wrapx, wrapy=wrapy)
        drawn = _plot_marked(ax, xs, ys, samples, (fmt,), {**kwargs, "label": "_nolegend_"})
        data_line = drawn[0] if drawn else None
        if data_line is not None:
            data_line.set_zorder(kwargs["zorder"] + 0.1)  # above the bars, as in ax.errorbar

    bar_color = (
        ecolor if ecolor is not None else (data_line.get_color() if data_line is not None else "C0")
    )
    if elinewidth is None:  # errorbar.elinewidth exists from matplotlib 3.11
        elinewidth = mpl.rcParams.get("errorbar.elinewidth")  # type: ignore[arg-type]
    bar_lw = elinewidth if elinewidth is not None else mpl.rcParams["lines.linewidth"]
    if capsize is None:
        capsize = mpl.rcParams["errorbar.capsize"]
    barlinecols: list[LineCollection] = []
    caplines: list[Line2D] = []

    def add_caps(cx: np.ndarray, cy: np.ndarray, marker: str) -> None:
        if capsize > 0:
            (cap,) = ax.plot(
                *wrap_points(cx, cy, wrapx=wrapx, wrapy=wrapy),
                linestyle="none",
                marker=marker,
                ms=2.0 * capsize,  # as in ax.errorbar, capsize is the half-width
                color=bar_color,
                markeredgecolor=bar_color,
            )
            caplines.append(cap)

    # x errors first, so the container's bars and caps come in ax.errorbar's order.
    for error, positions, values, horizontal, cap_marker in (
        (xerr, y, x, True, "|"),
        (yerr, x, y, False, "_"),
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
