from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path

import mpl_wrap
from mpl_wrap import (
    errorbar_wrapped,
    fill_between_wrapped,
    plot_wrapped,
    scatter_wrapped,
    set_wrap,
    stairs_wrapped,
    step_wrapped,
)
from mpl_wrap.plot import _to_num

WRAP360 = (0.0, 360.0)


def _arr(values: Any) -> np.ndarray:
    """Coerce loosely-typed artist data (get_xydata, path attributes) for numpy ops."""
    return np.asarray(values, dtype=float)


def _lines(artists: Any) -> list[Line2D]:
    """Narrow a helper's return to its lines (stairs_wrapped can also return a patch)."""
    assert isinstance(artists, list)
    return artists


def test_version() -> None:
    assert mpl_wrap.__version__ == "0.1.0"


def test_public_api() -> None:
    for name in (
        "set_wrap",
        "plot_wrapped",
        "scatter_wrapped",
        "fill_between_wrapped",
        "step_wrapped",
        "stairs_wrapped",
        "errorbar_wrapped",
    ):
        assert callable(getattr(mpl_wrap, name))


def test_to_num_none_passthrough() -> None:
    _, ax = plt.subplots()
    assert _to_num(ax.xaxis, None) is None


# plot_wrapped


def test_plot_wrapped_stays_in_window() -> None:
    _, ax = plt.subplots()
    x = np.linspace(0.0, 10.0, 200)
    (ln,) = plot_wrapped(ax, x, 100.0 * x, wrapy=WRAP360)
    ydata = _arr(ln.get_ydata())
    assert np.nanmin(ydata) >= 0.0 and np.nanmax(ydata) <= 360.0
    assert np.isnan(ydata).sum() > 0


def test_plot_wrapped_no_window_matches_plot() -> None:
    _, ax = plt.subplots()
    x = np.linspace(0.0, 10.0, 50)
    y = 100.0 * x
    (ln,) = plot_wrapped(ax, x, y)
    assert np.array_equal(_arr(ln.get_xdata()), x)
    assert np.array_equal(_arr(ln.get_ydata()), y)


def test_plot_wrapped_passes_style_args() -> None:
    _, ax = plt.subplots()
    (ln,) = plot_wrapped(ax, [0.0, 1.0], [0.0, 400.0], "r--", wrapy=WRAP360, label="a")
    assert ln.get_color() == "r"
    assert ln.get_label() == "a"


# scatter_wrapped


def test_scatter_wrapped_folds_points() -> None:
    _, ax = plt.subplots()
    pc = scatter_wrapped(ax, [0.5, 0.5, 0.5], [10.0, 370.0, 360.0], wrapy=WRAP360)
    offsets = _arr(pc.get_offsets())
    # 370 folds to 10, and a point exactly at the window max maps to the min
    assert np.allclose(offsets[:, 1], [10.0, 10.0, 0.0])
    assert np.allclose(offsets[:, 0], 0.5)


def test_scatter_wrapped_nan_passthrough() -> None:
    _, ax = plt.subplots()
    pc = scatter_wrapped(ax, [0.0, 1.0], [np.nan, 370.0], wrapy=WRAP360)
    offsets = _arr(pc.get_offsets())
    assert np.isnan(offsets[0, 1]) and np.allclose(offsets[1, 1], 10.0)


# fill_between_wrapped


def _moveto_count(codes: Any) -> int:
    return int((np.asarray(codes) == Path.MOVETO).sum())


def test_fill_between_tiles_y_offsets() -> None:
    _, ax = plt.subplots()
    x = np.array([0.0, 1.0])
    patch = fill_between_wrapped(ax, x, [0.0, 0.0], [10.0, 10.0], wrapy=WRAP360)
    path = patch.get_path()
    verts = _arr(path.vertices)
    # Narrow band tiles over period offsets -1..1: 3 tiles of 2*len(x) vertices
    assert _moveto_count(path.codes) == 3
    assert len(verts) == 3 * 2 * len(x)
    # Tiles are exact period-shifted copies covering the band at each offset
    ys = verts[:, 1].reshape(3, -1)
    assert np.allclose(ys[1] - ys[0], -360.0) or np.allclose(ys[1] - ys[0], 360.0)


def test_fill_between_saturated_band_collapses_to_rectangle() -> None:
    _, ax = plt.subplots()
    patch = fill_between_wrapped(ax, [0.0, 1.0], [0.0, 0.0], [400.0, 400.0], wrapy=WRAP360)
    path = patch.get_path()
    verts = _arr(path.vertices)
    # Band wider than the period fills the window with a single 4-vertex rectangle
    assert _moveto_count(path.codes) == 1
    assert len(verts) == 4
    assert set(np.unique(verts[:, 1])) == {0.0, 360.0}


def test_fill_between_mixed_saturation_bridges_runs() -> None:
    _, ax = plt.subplots()
    x = np.array([0.0, 1.0, 2.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([400.0, 100.0, 400.0])
    patch = fill_between_wrapped(ax, x, lo, hi, wrapy=WRAP360)
    # Two single-point saturated rectangles + the narrow run tiled over its offsets
    assert _moveto_count(patch.get_path().codes) == 2 + 3


def test_fill_between_both_axes_tiles_x_and_y() -> None:
    _, ax = plt.subplots()
    x = np.array([0.0, 1.0])
    patch = fill_between_wrapped(ax, x, [0.0, 0.0], [10.0, 10.0], wrapx=WRAP360, wrapy=WRAP360)
    # 3 x-offsets times 3 y-offsets
    assert _moveto_count(patch.get_path().codes) == 9


def test_fill_between_x_only_wrap() -> None:
    _, ax = plt.subplots()
    patch = fill_between_wrapped(ax, [350.0, 370.0], [0.0, 0.0], [1.0, 1.0], wrapx=WRAP360)
    # x spans 350..370, tiling over x period offsets -1..2
    assert _moveto_count(patch.get_path().codes) == 4
    # The window clip is applied (an axis-aligned rectangle becomes a clipbox)
    assert patch.clipbox is not None


def test_fill_between_empty_data() -> None:
    _, ax = plt.subplots()
    patch = fill_between_wrapped(ax, [], [], [], wrapy=WRAP360)
    assert len(_arr(patch.get_path().vertices)) == 0


def test_fill_between_where_and_interpolate_match_mpl() -> None:
    """Each run is built exactly as ax.fill_between builds its polygons."""
    _, ax = plt.subplots()
    x = np.linspace(0.0, 10.0, 21)
    y1, y2 = np.sin(x), np.cos(x)
    cases: tuple[tuple[Any, bool, Any], ...] = (
        (None, False, None),
        (y1 > y2, False, None),
        (y1 > y2, True, None),
        (None, False, "post"),
        (y1 > y2, True, "mid"),
    )
    for where, interpolate, step in cases:
        ref = ax.fill_between(x, y1, y2, where=where, interpolate=interpolate, step=step)
        patch = fill_between_wrapped(ax, x, y1, y2, where=where, interpolate=interpolate, step=step)
        ref_polys = [_arr(p.vertices) for p in ref.get_paths()]
        assert _moveto_count(patch.get_path().codes) == len(ref_polys)
        # Same point set per run (the band is traced in a different vertex order)
        got = _arr(patch.get_path().vertices)
        assert np.allclose(
            np.unique(np.round(got, 9), axis=0),
            np.unique(np.round(np.concatenate(ref_polys), 9), axis=0),
        )


def test_fill_between_runs_break_at_where_and_nonfinite() -> None:
    _, ax = plt.subplots()
    x = np.arange(6.0)
    hi = np.array([0.0, 400.0, 0.0, 0.0, 400.0, 0.0])
    # A False sample and a non-finite sample both end the run they fall in
    for lo, where in (
        (np.zeros(6), np.array([True, True, False, True, True, True])),
        (np.array([0.0, 0.0, np.nan, 0.0, 0.0, 0.0]), None),
    ):
        patch = fill_between_wrapped(ax, x, lo, hi, where=where, wrapy=WRAP360)
        verts = _arr(patch.get_path().vertices)
        assert np.isfinite(verts).all()
        assert verts[:, 0].min() == 0.0 and verts[:, 0].max() == 5.0
        assert _moveto_count(patch.get_path().codes) > 2  # a saturated run each side
        assert not np.any((verts[:, 0] > 1.0) & (verts[:, 0] < 3.0))  # the gap at x=2


def test_fill_between_no_window_single_tile() -> None:
    _, ax = plt.subplots()
    x = np.linspace(0.0, 1.0, 5)
    patch = fill_between_wrapped(ax, x, np.zeros(5), np.ones(5))
    path = patch.get_path()
    assert _moveto_count(path.codes) == 1
    assert len(_arr(path.vertices)) == 2 * len(x)


# step_wrapped


def test_step_wrapped_matches_mpl_and_wraps() -> None:
    _, ax = plt.subplots()
    x = np.array([0.0, 1.0, 2.0, 4.0])
    y = np.array([340.0, 380.0, 20.0, 30.0])
    for where in ("pre", "post", "mid"):
        # Unwrapped, the polyline is the one ax.step draws with drawstyle="steps-*"
        (ln,) = step_wrapped(ax, x, y, where=where)
        (ref,) = ax.step(x, y, where=where)
        assert np.allclose(_arr(ln.get_xydata()), _arr(ref.get_path().vertices))
        # Wrapped, risers crossing the seam are routed to the window edges
        (wrapped,) = step_wrapped(ax, x, y, where=where, wrapy=WRAP360)
        ydata = _arr(wrapped.get_ydata())
        assert np.nanmin(ydata) >= 0.0 and np.nanmax(ydata) <= 360.0
        assert np.isnan(ydata).sum() == 2  # up through 360, then back down
    (styled,) = step_wrapped(ax, x, y, "r--", wrapy=WRAP360, label="a")
    assert styled.get_color() == "r" and styled.get_label() == "a"


# stairs_wrapped


def test_stairs_wrapped_routes_riser_at_seam() -> None:
    _, ax = plt.subplots()
    (ln,) = _lines(
        stairs_wrapped(ax, [350.0, 370.0], [0.0, 1.0, 2.0], baseline=None, wrapy=WRAP360)
    )
    xdata, ydata = _arr(ln.get_xdata()), _arr(ln.get_ydata())
    assert np.nanmin(ydata) >= 0.0 and np.nanmax(ydata) <= 360.0
    assert np.isnan(ydata).sum() == 1
    # The riser at x=1 crosses the seam vertically
    seam = np.isfinite(ydata) & (xdata == 1.0)
    assert set(ydata[seam]) >= {0.0, 360.0}


def test_stairs_wrapped_default_edges() -> None:
    _, ax = plt.subplots()
    (ln,) = _lines(stairs_wrapped(ax, [10.0, 20.0, 30.0], wrapy=WRAP360))
    assert _arr(ln.get_xdata()).max() == 3.0


def test_stairs_wrapped_baselines() -> None:
    _, ax = plt.subplots()
    values, edges = [10.0, 20.0], [0.0, 1.0, 2.0]
    # As in ax.stairs, the default baseline=0 drops both ends down to it
    (drops,) = _lines(stairs_wrapped(ax, values, edges))
    assert _arr(drops.get_ydata())[[0, -1]].tolist() == [0.0, 0.0]
    (bare,) = _lines(stairs_wrapped(ax, values, edges, baseline=None))
    assert _arr(bare.get_ydata())[[0, -1]].tolist() == [10.0, 20.0]
    # An array baseline closes the outline back along its own staircase
    (closed,) = _lines(stairs_wrapped(ax, values, edges, baseline=[1.0, 2.0]))
    xy = np.column_stack([_arr(closed.get_xdata()), _arr(closed.get_ydata())])
    assert np.allclose(xy[0], xy[-1])
    # A baseline drop that crosses the seam is routed to the edges like any riser
    (wrapped,) = _lines(stairs_wrapped(ax, [350.0], [0.0, 1.0], baseline=-20.0, wrapy=WRAP360))
    ydata = _arr(wrapped.get_ydata())
    assert np.nanmin(ydata) >= 0.0 and np.nanmax(ydata) <= 360.0
    assert np.isnan(ydata).sum() == 2  # one break per end drop


def test_stairs_wrapped_horizontal_swaps_axes() -> None:
    _, ax = plt.subplots()
    (ln,) = _lines(
        stairs_wrapped(ax, [350.0, 370.0], [0.0, 1.0, 2.0], orientation="horizontal", wrapx=WRAP360)
    )
    xdata, ydata = _arr(ln.get_xdata()), _arr(ln.get_ydata())
    assert np.nanmin(xdata) >= 0.0 and np.nanmax(xdata) <= 360.0  # values wrap on x now
    assert np.nanmax(ydata) == 2.0  # edges run along y


def test_stairs_wrapped_fill_tiles_like_fill_between() -> None:
    _, ax = plt.subplots()
    values, edges = [10.0, 20.0], [0.0, 1.0, 2.0]
    patch = stairs_wrapped(ax, values, edges, fill=True, wrapy=WRAP360, alpha=0.3)
    assert isinstance(patch, PathPatch)
    assert patch in ax.patches
    verts = _arr(patch.get_path().vertices)
    assert verts[:, 1].min() < 0.0 and verts[:, 1].max() > 360.0  # tiled past the window
    assert patch.clipbox is not None  # clipped back to the window
    assert not patch.get_in_layout()
    # A horizontal fill tiles along x instead
    horiz = stairs_wrapped(ax, values, edges, orientation="horizontal", fill=True, wrapx=WRAP360)
    assert isinstance(horiz, PathPatch)
    hverts = _arr(horiz.get_path().vertices)
    assert hverts[:, 0].min() < 0.0 and hverts[:, 0].max() > 360.0


def test_wrapped_helpers_reject_bad_arguments() -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match=r"step_wrapped\(\) where='bogus'"):
        step_wrapped(ax, [0.0, 1.0], [0.0, 1.0], where="bogus")
    with pytest.raises(ValueError, match=r"fill_between_wrapped\(\) step='bogus'"):
        fill_between_wrapped(ax, [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], step="bogus")
    with pytest.raises(ValueError, match="where size"):
        fill_between_wrapped(ax, [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], where=[True])
    with pytest.raises(ValueError, match="cannot fill with baseline=None"):
        stairs_wrapped(ax, [10.0], [0.0, 1.0], baseline=None, fill=True)
    with pytest.raises(ValueError, match="orientation='sideways'"):
        stairs_wrapped(ax, [10.0], [0.0, 1.0], orientation="sideways")


# errorbar_wrapped


def test_errorbar_wrapped_container_structure() -> None:
    _, ax = plt.subplots()
    container = errorbar_wrapped(
        ax, [1.0], [355.0], yerr=[10.0], fmt="o", capsize=3, wrapy=WRAP360, label="pts"
    )
    assert isinstance(container, ErrorbarContainer)
    data_line, caplines, barlinecols = container.lines
    assert data_line is not None and np.allclose(_arr(data_line.get_ydata()), [355.0])
    assert len(caplines) == 2  # lo and hi caps
    assert len(barlinecols) == 1
    assert container.get_label() == "pts"
    assert container in ax.containers
    # Only the container is labeled, so the legend shows one entry (as in ax.errorbar)
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["pts"]


def test_errorbar_wrapped_bar_splits_at_seam() -> None:
    _, ax = plt.subplots()
    container = errorbar_wrapped(ax, [1.0], [355.0], yerr=[10.0], fmt="o", wrapy=WRAP360)
    segments = container.lines[2][0].get_segments()
    # The 345..365 bar splits into 345..360 and 0..5
    assert len(segments) == 2
    seg_ys = sorted(tuple(sorted(seg[:, 1])) for seg in segments)
    assert np.allclose(seg_ys[0], (0.0, 5.0))
    assert np.allclose(seg_ys[1], (345.0, 360.0))


def test_errorbar_wrapped_fmt_none_has_no_data_line() -> None:
    _, ax = plt.subplots()
    container = errorbar_wrapped(ax, [1.0], [355.0], yerr=[10.0], fmt="none", wrapy=WRAP360)
    assert container.lines[0] is None


def test_errorbar_wrapped_asymmetric_and_xerr() -> None:
    _, ax = plt.subplots()
    container = errorbar_wrapped(
        ax,
        [355.0],
        [5.0],
        xerr=np.array([[10.0], [7.0]]),
        fmt="o",
        wrapx=WRAP360,
    )
    assert container.has_xerr and not container.has_yerr
    segments = container.lines[2][0].get_segments()
    # The asymmetric 345..362 bar splits at the seam into 345..360 and 0..2
    assert len(segments) == 2
    seg_xs = sorted(tuple(sorted(seg[:, 0])) for seg in segments)
    assert np.allclose(seg_xs[0], (0.0, 2.0))
    assert np.allclose(seg_xs[1], (345.0, 360.0))


def test_errorbar_wrapped_all_nan_makes_no_segments() -> None:
    _, ax = plt.subplots()
    container = errorbar_wrapped(ax, [1.0], [np.nan], yerr=[1.0], fmt="o", wrapy=WRAP360)
    assert container.lines[2][0].get_segments() == []


# set_wrap


def test_set_wrap_stores_window_and_sets_lims() -> None:
    _, ax = plt.subplots()
    assert set_wrap(ax, wrapy=WRAP360) is ax  # returns the axes for chaining
    assert list(ax.lines) == []  # seam lines are opt-in
    assert ax.get_ylim() == (0.0, 360.0)  # limits set to the window

    (ln,) = plot_wrapped(ax, [0.0, 1.0], [350.0, 370.0])
    assert np.nanmax(_arr(ln.get_ydata())) <= 360.0


def test_set_wrap_no_lims() -> None:
    _, ax = plt.subplots()
    ylim = ax.get_ylim()
    set_wrap(ax, wrapy=WRAP360, set_lims=False)
    assert ax.get_ylim() == ylim


def test_set_wrap_seam_lines() -> None:
    _, ax = plt.subplots()
    set_wrap(ax, wrapy=WRAP360, seam_lines=True, seam_kwargs={"color": "C3"})
    lines = list(ax.lines)
    assert len(lines) == 2
    assert sorted(_arr(ln.get_ydata())[0] for ln in lines) == [0.0, 360.0]
    assert all(ln.get_color() == "C3" for ln in lines)


def test_wrap_true_requires_stored_window() -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="wrapy=True"):
        plot_wrapped(ax, [0.0, 1.0], [350.0, 370.0], wrapy=True)
    set_wrap(ax, wrapy=WRAP360)
    (ln,) = plot_wrapped(ax, [0.0, 1.0], [350.0, 370.0], wrapy=True)
    assert np.nanmax(_arr(ln.get_ydata())) <= 360.0


def test_set_wrap_rejects_true() -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="not valid in set_wrap"):
        set_wrap(ax, wrapy=True)


def test_set_wrap_explicit_kwarg_overrides_stored() -> None:
    _, ax = plt.subplots()
    set_wrap(ax, wrapy=WRAP360)
    (ln,) = plot_wrapped(ax, [0.0, 1.0], [350.0, 370.0], wrapy=(0.0, 720.0))
    ydata = _arr(ln.get_ydata())
    assert np.nanmax(ydata) == 370.0
    assert not np.isnan(ydata).any()


def test_set_wrap_false_disables_for_one_call() -> None:
    _, ax = plt.subplots()
    set_wrap(ax, wrapy=WRAP360)
    (ln,) = plot_wrapped(ax, [0.0, 1.0], [350.0, 370.0], wrapy=False)
    assert np.nanmax(_arr(ln.get_ydata())) == 370.0


def test_set_wrap_false_clears_stored_window() -> None:
    _, ax = plt.subplots()
    set_wrap(ax, wrapy=WRAP360)
    set_wrap(ax, wrapy=False)
    (ln,) = plot_wrapped(ax, [0.0, 1.0], [350.0, 370.0])
    assert np.nanmax(_arr(ln.get_ydata())) == 370.0


def test_set_wrap_updates_are_merged_per_axis() -> None:
    _, ax = plt.subplots()
    set_wrap(ax, wrapy=WRAP360)
    set_wrap(ax, wrapx=(0.0, 1.0))  # must not clear the stored y window
    (ln,) = plot_wrapped(ax, [0.0, 0.5], [350.0, 370.0])
    assert np.nanmax(_arr(ln.get_ydata())) <= 360.0


# datetimes


def test_datetime_data_and_window() -> None:
    _, ax = plt.subplots()
    t0 = np.datetime64("2026-01-01T00:00")
    times = t0 + np.arange(48) * np.timedelta64(1, "h")
    (ln,) = plot_wrapped(ax, times, np.arange(48.0), wrapx=(t0, t0 + np.timedelta64(1, "D")))
    x0 = float(ax.xaxis.convert_units(t0))
    xdata = _arr(ln.get_xdata())
    assert np.nanmin(xdata) >= x0 and np.nanmax(xdata) <= x0 + 1.0
    # The axis learned the date converter, so ticks format as dates
    assert ax.xaxis.get_converter() is not None


def test_datetime_window_in_set_wrap() -> None:
    _, ax = plt.subplots()
    t0 = np.datetime64("2026-01-01T00:00")
    set_wrap(ax, wrapx=(t0, t0 + np.timedelta64(1, "D")))
    times = t0 + np.arange(48) * np.timedelta64(1, "h")
    (ln,) = plot_wrapped(ax, times, np.arange(48.0))
    x0 = float(ax.xaxis.convert_units(t0))
    assert np.nanmax(_arr(ln.get_xdata())) <= x0 + 1.0


def test_datetime_wrap_y_of_numeric_angle() -> None:
    _, ax = plt.subplots()
    t0 = np.datetime64("2026-01-01T00:00")
    times = t0 + np.arange(100) * np.timedelta64(1, "h")
    (ln,) = plot_wrapped(ax, times, 30.0 * np.arange(100.0), wrapy=WRAP360)
    ydata = _arr(ln.get_ydata())
    assert np.nanmax(ydata) <= 360.0 and np.isnan(ydata).sum() > 0


# smoke tests


def test_smoke_render_all_helpers() -> None:
    fig, ax = plt.subplots()
    set_wrap(ax, wrapy=WRAP360)
    x = np.linspace(0.0, 10.0, 100)
    plot_wrapped(ax, x, 100.0 * x, label="line")
    scatter_wrapped(ax, x[::10], 100.0 * x[::10], label="points")
    fill_between_wrapped(ax, x, 90.0 * x, 110.0 * x, alpha=0.3, label="band")
    step_wrapped(ax, x, 100.0 * x, where="post")
    stairs_wrapped(ax, 100.0 * x[:-1], x)
    stairs_wrapped(ax, 100.0 * x[:-1], x, fill=True, alpha=0.3)
    errorbar_wrapped(ax, x[::20], 100.0 * x[::20], yerr=20.0 + x[::20], fmt="o", capsize=2)
    ax.legend(loc="upper left")
    fig.canvas.draw()


def test_smoke_render_wrap_both_axes() -> None:
    fig, ax = plt.subplots()
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    window = (-1.0, 1.0)
    plot_wrapped(ax, 1.2 * np.cos(theta), 1.2 * np.sin(theta), wrapx=window, wrapy=window)
    x_fill = np.linspace(-1.2, 1.2, 100)
    semi = np.sqrt(1.2**2 - x_fill**2)
    fill_between_wrapped(ax, x_fill, -semi, semi, wrapx=window, wrapy=window, alpha=0.3)
    fig.canvas.draw()
