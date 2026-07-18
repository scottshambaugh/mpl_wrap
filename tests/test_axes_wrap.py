import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mpl_wrap import AxesWrap, AxesWrapBase, wrap_axes

WRAP360 = (0.0, 360.0)


def _arr(values: Any) -> np.ndarray:
    """Coerce loosely-typed artist data (get_xydata, path attributes) for numpy ops."""
    return np.asarray(values, dtype=float)


def test_axeswrap_projection() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": "wrap"})
    assert isinstance(ax, AxesWrap)
    assert ax.set_wrap(wrapy=WRAP360) is ax  # returns self for chaining
    (ln,) = ax.plot_wrapped([0.0, 1.0], [350.0, 370.0])
    assert np.nanmax(_arr(ln.get_ydata())) <= 360.0


def test_wrap_axes_upgrades_in_place_and_mixes() -> None:
    _, ax = plt.subplots()
    axw = wrap_axes(ax, wrapy=WRAP360)
    assert axw is ax
    assert isinstance(ax, AxesWrap)
    (wrapped,) = axw.plot_wrapped([0.0, 1.0], [350.0, 370.0])
    (plain,) = axw.plot([0.0, 1.0], [350.0, 370.0])  # inherited methods untouched
    assert np.nanmax(_arr(wrapped.get_ydata())) <= 360.0
    assert np.nanmax(_arr(plain.get_ydata())) == 370.0


def test_axeswrap_methods_delegate() -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "wrap"})
    assert isinstance(ax, AxesWrap)
    ax.set_wrap(wrapy=WRAP360)
    x = np.linspace(0.0, 10.0, 100)
    ax.scatter_wrapped(x[::10], 100.0 * x[::10])
    ax.fill_between_wrapped(x, 90.0 * x, 110.0 * x, alpha=0.3)
    ax.stairs_wrapped(100.0 * x[:-1], x)
    ax.errorbar_wrapped(x[::20], 100.0 * x[::20], yerr=20.0 + x[::20], fmt="o")
    fig.canvas.draw()


def test_axeswrap_data_methods_use_stored_window() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": "wrap"})
    assert isinstance(ax, AxesWrap)
    ax.set_wrap(wrapy=WRAP360)
    xs, ys = ax.wrap_line([0.0, 1.0], [350.0, 370.0])
    assert np.nanmax(ys) <= 360.0
    assert np.isnan(ys).sum() == 1
    px, py = ax.wrap_points([0.5], [370.0])
    assert np.allclose(px, [0.5]) and np.allclose(py, [10.0])


def test_wrap_axes_other_projection_and_idempotent() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    axw = wrap_axes(ax)
    cls = type(axw)
    assert cls.__name__ == "AxesWrapPolarAxes"
    assert isinstance(axw, AxesWrapBase)
    assert not isinstance(axw, AxesWrap)  # keeps its polar base, not rectilinear
    wrap_axes(axw)
    assert type(axw) is cls  # idempotent: not wrapped twice


def test_axeswrap_pickle_roundtrip() -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "wrap"})
    assert isinstance(ax, AxesWrap)
    ax.set_wrap(wrapy=WRAP360)
    ax.plot_wrapped([0.0, 1.0], [350.0, 370.0])

    fig2 = pickle.loads(pickle.dumps(fig))
    ax2 = fig2.axes[0]
    assert isinstance(ax2, AxesWrap)
    (ln,) = ax2.plot_wrapped([0.0, 1.0], [350.0, 370.0])  # stored window survives
    assert np.nanmax(_arr(ln.get_ydata())) <= 360.0


def test_wrap_axes_pickle_roundtrip_other_projection() -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    wrap_axes(ax, wrapy=(0.0, 1.0), set_lims=False)

    fig2 = pickle.loads(pickle.dumps(fig))
    ax2 = fig2.axes[0]
    assert type(ax2).__name__ == "AxesWrapPolarAxes"
    assert isinstance(ax2, AxesWrapBase)
    (ln,) = ax2.plot_wrapped([0.0, 1.0], [0.5, 1.5])
    assert np.nanmax(_arr(ln.get_ydata())) <= 1.0
