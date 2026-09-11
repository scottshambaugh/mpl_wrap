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
from matplotlib.ticker import MultipleLocator

from mpl_wrap import (
    fill_between_wrapped,
    plot_wrapped,
    set_wrap,
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


def pi_demo(savedir: Path = SAVEDIR) -> None:
    """A phase in radians on a (-pi, pi) window: pi/2 ticks labelled as fractions of pi."""
    t = np.linspace(0, 10, 500)
    phase = 1.5 * t  # radians
    width = 0.1 + 0.07 * t

    fig, ax = plt.subplots(figsize=(6, 3))
    set_wrap(ax, wrapy=(-np.pi, np.pi))  # a radian window: pi/2 ticks, pi labels
    fill_between_wrapped(ax, t, phase - width, phase + width, alpha=0.3, label="uncertainty")
    plot_wrapped(ax, t, phase, label="phase")
    ax.set(xlabel="time (s)", ylabel="phase (rad)")
    ax.legend()

    _save_demo(fig, savedir, "pi_demo.png")


def wrapy_demo(savedir: Path = SAVEDIR) -> None:
    """Unwrapped vs naive modulo vs mpl_wrap comparison of an angle winding up."""
    wrapy = (0.0, 360.0)
    period = wrapy[1] - wrapy[0]

    # An angle that winds up, reverses at t = 25, and unwinds, with a band growing
    # past a full period, and a data gap over t = 40..60 spanning several periods.
    reverse = 25.0
    x = np.concatenate([np.linspace(0.0, 40.0, 40), np.linspace(60.0, 70.0, 10)])
    center = 40.0 * np.where(x <= reverse, x, 2.0 * reverse - x)  # unwrapped angle, deg
    half_width = 2.5 + 4.0 * x
    lower = center - half_width
    upper = center + half_width

    # Shared styles, so the panels differ only in how the data is projected.
    band_style: dict[str, Any] = {"color": "C0", "alpha": 0.4, "label": "fill_between"}
    line_style: dict[str, Any] = {"color": "C0", "label": "plot", "marker": "*"}

    fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for index, ax in enumerate(axs):
        ax.set(xlim=(x[0], x[-1]), ylabel="angle (deg)")
        ax.grid(True, alpha=0.3)
        if index > 0:
            set_wrap(ax, wrapy=wrapy, seam_lines=True)
            # Pad past the window so the seam lines and edge routing are visible.
            ax.set_ylim(wrapy[0] - 0.05 * period, wrapy[1] + 0.05 * period)

    axs[0].set_title("Unwrapped")
    axs[0].yaxis.set_major_locator(MultipleLocator(period))
    axs[0].fill_between(x, lower, upper, **band_style)
    axs[0].plot(x, center, **line_style)

    axs[1].set_title("Modulus (y % 360)")
    axs[1].fill_between(x, lower % period, upper % period, **band_style)
    axs[1].plot(x, center % period, **line_style)

    axs[2].set_title("mpl_wrap")
    fill_between_wrapped(axs[2], x, lower, upper, **band_style)
    plot_wrapped(axs[2], x, center, **line_style)

    axs[0].legend(loc="upper left")
    _save_demo(fig, savedir, "wrapy_demo.png")


def circle_demo(savedir: Path = SAVEDIR) -> None:
    """Plot a radius-1.2 disk on a 2x2 grid of x/y wrapping combinations.

    The disk's bottom-right quadrant is squared off to a corner, so the shape is
    symmetric across neither axis and each panel wraps it differently.
    """
    window = (-1, 1)
    radius = 1.2
    # Outline: three quadrants of the circle, closed by the squared-off corner.
    theta = np.linspace(0.0, 1.5 * np.pi, 300)
    x = np.concatenate([radius * np.cos(theta), [radius, radius]])
    y = np.concatenate([radius * np.sin(theta), [-radius, 0.0]])
    # The disk as a fill between the lower and upper boundaries.
    x_fill = np.linspace(-radius, radius, 400)
    semi = np.sqrt(radius**2 - x_fill**2)
    lower = np.where(x_fill >= 0.0, -radius, -semi)
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
            fill_between_wrapped(ax, x_fill, lower, semi, color="C0", alpha=0.25)
            plot_wrapped(ax, x, y, color="C0")
            ax.yaxis.set_major_locator(MultipleLocator(window[1]))
            ax.xaxis.set_major_locator(MultipleLocator(window[1]))
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


def geo_demo(savedir: Path = SAVEDIR) -> None:
    """The README geographic plot: a ground track over the poles and the dateline."""
    try:
        import cartopy.crs as ccrs
    except ImportError:
        print("Skipped geo_demo: cartopy is not installed")
        return

    from mpl_wrap import fill_around, scatter_wrapped

    u = np.linspace(0, 720, 400)  # two turns round the orbit, in degrees
    lon = -160 + 0.3 * u  # drifts east, past the antimeridian
    lat = u  # runs past both poles

    fig, ax = plt.subplots(figsize=(6, 3.4), subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    ax.coastlines(linewidth=0.4, color="0.55")
    ax.gridlines(linewidth=0.3, color="0.9")
    fill_around(ax, lon, lat, 5, facecolor="C0", alpha=0.3, label="5 deg corridor")
    plot_wrapped(ax, lon, lat, color="C0", linewidth=1.4, label="ground track")
    scatter_wrapped(ax, lon[::40], lat[::40], s=14, color="C0", zorder=3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    _save_demo(fig, savedir, "geo_demo.png")


def geo_projections_demo(savedir: Path = SAVEDIR) -> None:
    """A projection grid: the same ground covered whatever the projection.

    A polar orbit's latitude argument runs past the poles rather than turning
    around at them, and its longitude runs past the antimeridian, so every
    helper here is fed continuous data well outside the usual lat/lon ranges.
    The track is timed so that one pole crossing lands exactly on the dateline -
    the corner of the lon/lat domain, where both seams have to be handled at once.
    """
    try:
        import cartopy.crs as ccrs
    except ImportError:
        print("Skipped geo_projections_demo: cartopy is not installed")
        return

    from mpl_wrap import fill_around, scatter_wrapped

    # Argument of latitude: the angle around the orbit, which for a polar orbit
    # is the latitude itself once folded at the poles.
    u = np.linspace(0.0, 1080.0, 3000)
    lat = u
    # Longitude drifts westward as the Earth turns beneath the orbit. The offset
    # puts the pole crossing at u = 450 exactly on the antimeridian.
    lon = -180.0 + 0.4 * (u - 450.0)
    # Two bands on the track, for contrast. Orange is fill_around: a constant
    # width on the ellipsoid, which holds over a pole and only looks wide there
    # because the projection stretches longitude. Blue is fill_between_wrapped:
    # an offset in degrees of latitude, which narrows to nothing at the pole.
    lat_width = 12.0  # degrees of latitude, above and below the track
    corridor_width = 2.7  # degrees of arc, perpendicular to the track

    panels = (
        ("PlateCarree", ccrs.PlateCarree(), None),
        ("Mollweide", ccrs.Mollweide(), None),
        ("Robinson", ccrs.Robinson(), None),
        ("Mercator", ccrs.Mercator(), None),
        ("Orthographic", ccrs.Orthographic(-40, 35), None),
        ("NorthPolarStereo", ccrs.NorthPolarStereo(), [-180, 180, 25, 90]),
        ("SouthPolarStereo", ccrs.SouthPolarStereo(), [-180, 180, -90, -25]),
        ("InterruptedGoodeHomolosine", ccrs.InterruptedGoodeHomolosine(), None),
        ("Geostationary", ccrs.Geostationary(), None),
    )
    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig = plt.figure(figsize=(4.6 * ncols, 3.1 * nrows))
    for i, (title, proj, extent) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=proj)
        if extent is None:
            ax.set_global()
        else:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4, color="0.55")
        ax.gridlines(linewidth=0.3, color="0.85")
        # A GeoAxes is geographic by default, so the helpers fold at the poles
        # and wrap longitude at the antimeridian without being told to.
        set_wrap(ax, set_lims=False)
        # The constant-width corridor goes down first, under the latitude band.
        fill_around(ax, lon, lat, corridor_width, facecolor="#f6c28b", zorder=1)
        fill_between_wrapped(
            ax, lon, lat - lat_width, lat + lat_width, color="C0", alpha=0.22, zorder=2
        )
        plot_wrapped(ax, lon, lat, color="C0", linewidth=1.1, zorder=3)
        scatter_wrapped(ax, lon[::150], lat[::150], s=9, color="C3", zorder=4)
        ax.set_title(title, fontsize=9)

    _save_demo(fig, savedir, "geo_projections.png")


if __name__ == "__main__":
    basic_usage_demo()
    pi_demo()
    wrapy_demo()
    circle_demo()
    datetime_demo()
    geo_demo()
    geo_projections_demo()
