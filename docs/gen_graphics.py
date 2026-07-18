"""Regenerate the demo graphics embedded in the README.

Run with: uv run python docs/gen_graphics.py
"""

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from mpl_wrap import (
    errorbar_wrapped,
    fill_between_wrapped,
    plot_wrapped,
    set_wrap,
    stairs_wrapped,
)

SAVEDIR = Path(__file__).parent


def _save_demo(fig: plt.Figure, savedir: Path, filename: str) -> None:
    """Lay out, save, close, and report a generated demo figure."""
    path = savedir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"Saved {path}")


def basic_usage_demo(savedir: Path = SAVEDIR) -> None:
    """The README Basic Usage plot: a precessing angle with uncertainty on a wrapped axis."""
    t = np.linspace(0, 10, 500)
    angle = 80.0 * t  # degrees
    width = 5.0 + 4.0 * t

    fig, ax = plt.subplots(figsize=(6, 3))
    set_wrap(ax, wrapy=(0, 360))  # helpers on ax now wrap y into (0, 360)
    fill_between_wrapped(ax, t, angle - width, angle + width, alpha=0.3, label="uncertainty")
    plot_wrapped(ax, t, angle, label="angle")
    ax.set(xlabel="time (s)", ylabel="angle (deg)")
    ax.legend()

    _save_demo(fig, savedir, "basic_usage.png")


def wrapy_demo(savedir: Path = SAVEDIR) -> None:
    """Unwrapped vs naive modulo vs mpl_wrap comparison of an angle winding up."""
    wrapy = (0.0, 360.0)
    period = wrapy[1] - wrapy[0]

    # An angle that winds up, reverses at t = 25, and unwinds, with a band growing
    # past a full period, and a data gap over t = 40..60 spanning several periods.
    reverse = 25.0
    x = np.concatenate([np.linspace(0.0, 40.0, 400), np.linspace(60.0, 70.0, 100)])
    center = 40.0 * np.where(x <= reverse, x, 2.0 * reverse - x)  # unwrapped angle, deg
    half_width = 2.5 + 6.0 * x
    lower = center - half_width
    upper = center + half_width

    # A coarse step series over the same span. The gap is one wide bridging bin.
    edges = np.concatenate([np.linspace(0.0, 40.0, 21), np.linspace(60.0, 70.0, 6)])
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = 40.0 * np.where(centers <= reverse, centers, 2.0 * reverse - centers)

    # Discrete measurements with growing error bars, excluding the gap bin.
    measured = np.diff(edges) < 10.0
    err_x = centers[measured]
    err_y = values[measured]
    err_yerr = (1.25 + 3.0 * centers)[measured]

    # Shared styles, so the panels differ only in how the data is projected.
    band_style: dict[str, Any] = {"color": "C0", "alpha": 0.4, "label": "fill_between"}
    line_style: dict[str, Any] = {"color": "C0", "label": "plot"}
    stairs_style: dict[str, Any] = {"color": "C1", "label": "stairs"}
    err_style: dict[str, Any] = {"fmt": "o", "ms": 3, "color": "C2", "label": "errorbar"}

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for index, ax in enumerate(axs):
        ax.set(xlim=(x[0], x[-1]), ylabel="angle (deg)")
        ax.grid(True, alpha=0.3)
        if index > 0:
            set_wrap(ax, wrapy=wrapy, seam_lines=True)
            # Pad past the window so the seam lines and edge routing are visible.
            ax.set_ylim(wrapy[0] - 0.05 * period, wrapy[1] + 0.05 * period)

    axs[0].set_title("Unwrapped")
    axs[0].fill_between(x, lower, upper, **band_style)
    axs[0].plot(x, center, **line_style)
    axs[0].stairs(values, edges, baseline=None, **stairs_style)
    axs[0].errorbar(err_x, err_y, err_yerr, **err_style)

    axs[1].set_title("mpl_wrap")
    fill_between_wrapped(axs[1], x, lower, upper, **band_style)
    plot_wrapped(axs[1], x, center, **line_style)
    stairs_wrapped(axs[1], values, edges, **stairs_style)
    errorbar_wrapped(axs[1], err_x, err_y, err_yerr, **err_style)

    axs[2].set_title("Modulus (y % 360)")
    axs[2].fill_between(x, lower % period, upper % period, **band_style)
    axs[2].plot(x, center % period, **line_style)
    axs[2].stairs(values % period, edges, baseline=None, **stairs_style)
    axs[2].errorbar(err_x, err_y % period, err_yerr, **err_style)

    axs[0].legend(loc="upper left")
    _save_demo(fig, savedir, "wrapy_demo.png")


def circle_demo(savedir: Path = SAVEDIR) -> None:
    """Plot a radius-1.2 circle on a 2x2 grid of x/y wrapping combinations."""
    window = (-1, 1)
    radius = 1.2
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    # The disk as a fill between the lower and upper semicircles.
    x_fill = np.linspace(-radius, radius, 400)
    semi = np.sqrt(radius**2 - x_fill**2)
    pad = 1.35

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    # Columns wrap x, rows wrap y.
    for row, wrapy in enumerate([None, window]):
        for col, wrapx in enumerate([None, window]):
            ax = axs[row, col]
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"wrapx={wrapx}, wrapy={wrapy}")
            set_wrap(ax, wrapx=wrapx, wrapy=wrapy, set_lims=False, seam_lines=True)
            fill_between_wrapped(ax, x_fill, -semi, semi, color="C0", alpha=0.25)
            plot_wrapped(ax, x, y, color="C0")
            ax.set(xlim=(-pad, pad), ylim=(-pad, pad))

    _save_demo(fig, savedir, "circle_demo.png")


def _colored_line(
    ax: Axes, x: np.ndarray, y: np.ndarray, c: np.ndarray, norm: Normalize
) -> LineCollection:
    """Add a polyline colored smoothly by ``c`` (viridis), skipping NaN breaks."""
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = 0.5 * (c[:-1] + c[1:])
    valid = ~np.isnan(segments).any(axis=(1, 2))
    lc = LineCollection(segments[valid].tolist(), cmap="viridis", norm=norm, linewidth=1.5)
    lc.set_array(colors[valid])
    ax.add_collection(lc)
    ax.autoscale_view()
    return lc


def datetime_demo(savedir: Path = SAVEDIR) -> None:
    """Fold a multi-day signal onto a single day, with the line colored by datetime."""
    day = np.timedelta64(1, "D")
    t0 = np.datetime64("2026-01-01T00:00")
    minutes = np.arange(0, 5 * 24 * 60, 15)  # 5 days at 15-minute cadence
    times = t0 + minutes * np.timedelta64(1, "m")
    hours = minutes / 60.0
    # A diurnal signal with a day-to-day drift.
    signal = np.sin(2.0 * np.pi * hours / 24.0) + 0.15 * (hours / 24.0)

    tnum = mdates.date2num(times)
    norm = Normalize(tnum[0], tnum[-1])

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    # Top: the unwrapped series over the full five days, colored by datetime.
    axs[0].set(title="Unwrapped", ylabel="signal")
    axs[0].xaxis_date()
    _colored_line(axs[0], tnum, signal, tnum, norm)
    locator = mdates.AutoDateLocator()
    axs[0].xaxis.set_major_locator(locator)
    axs[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # Bottom: the same series wrapped onto one day, so the days overlay as a
    # time-of-day view. The shared coloring shows which day is which.
    axs[1].set(title="Wrapped onto one day", ylabel="signal", xlabel="time of day")
    set_wrap(axs[1], wrapx=(t0, t0 + day))
    (line,) = plot_wrapped(axs[1], times, signal)
    xw = np.asarray(line.get_xdata(), dtype=float)
    yw = np.asarray(line.get_ydata(), dtype=float)
    line.remove()
    # Recover the continuous datetime at each wrapped vertex: each NaN break is
    # one seam crossing, i.e. one more day folded back (1 day = 1.0 date units).
    cw = xw + np.cumsum(np.isnan(xw))
    _colored_line(axs[1], xw, yw, cw, norm)
    axs[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for ax in axs:
        ax.grid(True, alpha=0.3)
    _save_demo(fig, savedir, "datetime_demo.png")


if __name__ == "__main__":
    basic_usage_demo()
    wrapy_demo()
    circle_demo()
    datetime_demo()
