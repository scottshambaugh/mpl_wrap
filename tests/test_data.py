import numpy as np

from mpl_wrap import wrap_line, wrap_points
from mpl_wrap.data import _wrap_polyline
from mpl_wrap.geo import fold_poles

WRAP360 = (0.0, 360.0)


# _wrap_polyline


def test_polyline_no_crossing_folds_into_window() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([365.0, 370.0, 375.0])  # one period up, no boundary crossed
    out_x, out_y, _ = _wrap_polyline(x, y, np.array(WRAP360))
    assert np.array_equal(out_x, x)
    assert np.allclose(out_y, [5.0, 10.0, 15.0])


def test_polyline_ascending_crossing_routes_to_edges() -> None:
    out_x, out_y, _ = _wrap_polyline(
        np.array([0.0, 1.0]), np.array([350.0, 370.0]), np.array(WRAP360)
    )
    # sample, exit at top edge, NaN break, re-enter at bottom edge, sample
    assert np.allclose(out_x, [0.0, 0.5, 0.5, 0.5, 1.0])
    assert np.allclose(out_y[[0, 1, 3, 4]], [350.0, 360.0, 0.0, 10.0])
    assert np.isnan(out_y[2])


def test_polyline_descending_crossing_routes_to_edges() -> None:
    out_x, out_y, _ = _wrap_polyline(
        np.array([0.0, 1.0]), np.array([10.0, -10.0]), np.array(WRAP360)
    )
    assert np.allclose(out_x, [0.0, 0.5, 0.5, 0.5, 1.0])
    assert np.allclose(out_y[[0, 1, 3, 4]], [10.0, 0.0, 360.0, 350.0])
    assert np.isnan(out_y[2])


def test_polyline_crossing_interpolates_at_correct_slope() -> None:
    # Rises 40/unit from y=340: crosses 360 at x = 0.5
    out_x, out_y, _ = _wrap_polyline(
        np.array([0.0, 1.0]), np.array([340.0, 380.0]), np.array(WRAP360)
    )
    assert np.allclose(out_x[1], 0.5)
    # Segments on both sides of the seam have the same slope as the input
    assert np.allclose((out_y[1] - out_y[0]) / (out_x[1] - out_x[0]), 40.0)
    assert np.allclose((out_y[4] - out_y[3]) / (out_x[4] - out_x[3]), 40.0)


def test_polyline_multi_period_segment_sweeps_window_each_period() -> None:
    out_x, out_y, _ = _wrap_polyline(
        np.array([0.0, 1.0]), np.array([10.0, 730.0]), np.array(WRAP360)
    )
    # 2 samples + 2 crossings x 3 vertices
    assert len(out_x) == 8
    finite = np.isfinite(out_y)
    assert np.isnan(out_y).sum() == 2
    assert out_y[finite].min() == 0.0 and out_y[finite].max() == 360.0
    # Crossing x positions interpolate the 360 and 720 boundaries
    assert np.allclose(out_x[1], (360.0 - 10.0) / 720.0)
    assert np.allclose(out_x[4], (720.0 - 10.0) / 720.0)


def test_polyline_nonfinite_input_passes_through_as_break() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, np.nan, 20.0, 30.0])
    out_x, out_y, _ = _wrap_polyline(x, y, np.array(WRAP360))
    assert np.array_equal(out_x, x)
    assert np.isnan(out_y[1])
    assert np.allclose(out_y[[0, 2, 3]], [10.0, 20.0, 30.0])


def test_polyline_empty_input() -> None:
    out_x, out_y, samples = _wrap_polyline(np.array([]), np.array([]), np.array(WRAP360))
    assert len(out_x) == 0 and len(out_y) == 0 and len(samples) == 0


def test_polyline_reports_where_samples_landed() -> None:
    x, y = np.array([0.0, 1.0, 2.0]), np.array([350.0, 370.0, 380.0])
    out_x, out_y, samples = _wrap_polyline(x, y, np.array(WRAP360))
    # The crossing inserts three vertices, so the later samples shift along
    assert samples.tolist() == [0, 4, 5]
    assert np.allclose(out_x[samples], x)
    assert np.allclose(out_y[samples], [350.0, 10.0, 20.0])


# wrap_line


def test_wrap_line_wraps_x_via_swap() -> None:
    xs, ys = wrap_line([350.0, 370.0], [0.0, 1.0], wrapx=WRAP360)
    assert np.allclose(xs[[0, 1, 3, 4]], [350.0, 360.0, 0.0, 10.0])
    assert np.isnan(xs[2])
    assert np.allclose(ys[[0, 1, 3, 4]], [0.0, 0.5, 0.5, 1.0])


def test_wrap_line_composes_both_axes() -> None:
    xs, ys = wrap_line([350.0, 370.0], [350.0, 370.0], wrapx=WRAP360, wrapy=WRAP360)
    finite = np.isfinite(xs) & np.isfinite(ys)
    assert xs[finite].min() >= 0.0 and xs[finite].max() <= 360.0
    assert ys[finite].min() >= 0.0 and ys[finite].max() <= 360.0


# wrap_points


def test_wrap_points_folds_pointwise() -> None:
    xs, ys = wrap_points([0.5, 1.5, 2.5], [10.0, 370.0, np.nan], wrapy=WRAP360)
    assert np.allclose(xs, [0.5, 1.5, 2.5])  # x untouched without a wrapx window
    assert np.allclose(ys[:2], [10.0, 10.0])
    assert np.isnan(ys[2])


# geographic pole folding (no cartopy needed)


def test_fold_poles_reflects_past_the_pole() -> None:
    lat = np.array([0.0, 45.0, 100.0, 180.0, 200.0, 280.0, -100.0, -90.0])
    lon, out = fold_poles(np.zeros_like(lat), lat)
    assert np.allclose(out, [0.0, 45.0, 80.0, 0.0, -20.0, -80.0, -80.0, -90.0])
    # A reflected latitude moves to the antipodal meridian, an in-range one does not.
    assert np.allclose(lon, [0.0, 0.0, 180.0, 180.0, 180.0, 0.0, 180.0, 0.0])


def test_fold_poles_leaves_in_range_latitudes_alone() -> None:
    lat = np.array([-90.0, -30.0, 0.0, 30.0, 89.0])
    lon = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    out_lon, out_lat = fold_poles(lon, lat)
    assert np.array_equal(out_lat, lat)
    assert np.array_equal(out_lon, lon)


def test_wrap_line_geographic_routes_over_the_pole() -> None:
    xs, ys = wrap_line([10.0, 10.0], [80.0, 100.0], wrapx=(-180.0, 180.0), geographic=True)
    # up to the pole on the original meridian, a break, then down the antipodal one
    assert np.allclose(xs[[0, 1]], [10.0, 10.0])
    assert np.allclose(ys[[0, 1]], [80.0, 90.0])
    assert np.isnan(xs[2]) and np.isnan(ys[2])
    assert np.allclose(xs[[3, 4]], [-170.0, -170.0])
    assert np.allclose(ys[[3, 4]], [90.0, 80.0])


def test_wrap_line_geographic_ignores_wrapy() -> None:
    args = ([0.0, 0.0], [80.0, 100.0])
    a = wrap_line(*args, geographic=True)
    b = wrap_line(*args, wrapy=(-90.0, 90.0), geographic=True)
    assert np.allclose(a[0], b[0], equal_nan=True)
    assert np.allclose(a[1], b[1], equal_nan=True)


def test_geographic_longitude_window_defaults_to_the_globe() -> None:
    xs, _ = wrap_line([170.0, 190.0], [0.0, 0.0], geographic=True)
    assert np.allclose(xs[[0, -1]], [170.0, -170.0])
    xs, _ = wrap_points([190.0], [0.0], geographic=True)
    assert np.allclose(xs, [-170.0])


def test_wrap_points_geographic_folds_and_wraps_longitude() -> None:
    xs, ys = wrap_points([170.0], [100.0], wrapx=(-180.0, 180.0), geographic=True)
    # 170 + 180 = 350, which folds back to -10
    assert np.allclose(xs, [-10.0])
    assert np.allclose(ys, [80.0])


def test_wrap_line_return_samples_indexes_the_input_points() -> None:
    x = [0.0, 1.0, 2.0]
    y = [350.0, 370.0, 390.0]  # one seam crossing between the first two samples
    xs, ys, samples = wrap_line(x, y, wrapy=WRAP360, return_samples=True)
    assert len(samples) == 3
    assert np.allclose(xs[samples], x)
    assert np.allclose(ys[samples], [350.0, 10.0, 30.0])
    assert len(xs) == 6  # the crossing inserted three vertices
    assert len(wrap_line(x, y, wrapy=WRAP360)) == 2  # the default is unchanged
