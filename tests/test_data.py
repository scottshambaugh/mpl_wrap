import numpy as np

from mpl_wrap import wrap_line, wrap_points
from mpl_wrap.data import _wrap_polyline

WRAP360 = (0.0, 360.0)


# _wrap_polyline


def test_polyline_no_crossing_folds_into_window() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([365.0, 370.0, 375.0])  # one period up, no boundary crossed
    out_x, out_y = _wrap_polyline(x, y, np.array(WRAP360))
    assert np.array_equal(out_x, x)
    assert np.allclose(out_y, [5.0, 10.0, 15.0])


def test_polyline_ascending_crossing_routes_to_edges() -> None:
    out_x, out_y = _wrap_polyline(np.array([0.0, 1.0]), np.array([350.0, 370.0]), np.array(WRAP360))
    # sample, exit at top edge, NaN break, re-enter at bottom edge, sample
    assert np.allclose(out_x, [0.0, 0.5, 0.5, 0.5, 1.0])
    assert np.allclose(out_y[[0, 1, 3, 4]], [350.0, 360.0, 0.0, 10.0])
    assert np.isnan(out_y[2])


def test_polyline_descending_crossing_routes_to_edges() -> None:
    out_x, out_y = _wrap_polyline(np.array([0.0, 1.0]), np.array([10.0, -10.0]), np.array(WRAP360))
    assert np.allclose(out_x, [0.0, 0.5, 0.5, 0.5, 1.0])
    assert np.allclose(out_y[[0, 1, 3, 4]], [10.0, 0.0, 360.0, 350.0])
    assert np.isnan(out_y[2])


def test_polyline_crossing_interpolates_at_correct_slope() -> None:
    # Rises 40/unit from y=340: crosses 360 at x = 0.5
    out_x, out_y = _wrap_polyline(np.array([0.0, 1.0]), np.array([340.0, 380.0]), np.array(WRAP360))
    assert np.allclose(out_x[1], 0.5)
    # Segments on both sides of the seam have the same slope as the input
    assert np.allclose((out_y[1] - out_y[0]) / (out_x[1] - out_x[0]), 40.0)
    assert np.allclose((out_y[4] - out_y[3]) / (out_x[4] - out_x[3]), 40.0)


def test_polyline_multi_period_segment_sweeps_window_each_period() -> None:
    out_x, out_y = _wrap_polyline(np.array([0.0, 1.0]), np.array([10.0, 730.0]), np.array(WRAP360))
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
    out_x, out_y = _wrap_polyline(x, y, np.array(WRAP360))
    assert np.array_equal(out_x, x)
    assert np.isnan(out_y[1])
    assert np.allclose(out_y[[0, 2, 3]], [10.0, 20.0, 30.0])


def test_polyline_empty_input() -> None:
    out_x, out_y = _wrap_polyline(np.array([]), np.array([]), np.array(WRAP360))
    assert len(out_x) == 0 and len(out_y) == 0


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
