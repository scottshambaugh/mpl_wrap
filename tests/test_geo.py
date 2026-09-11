"""Geographic support on cartopy axes.

Skipped whole-file when cartopy is not installed: it is a test-only dependency,
and mpl_wrap must never import it (see test_plot.test_import_does_not_pull_in_cartopy).
"""

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpl_wrap import (
    axhspan_wrapped,
    axvspan_wrapped,
    errorbar_wrapped,
    fill_around,
    fill_between_wrapped,
    hlines_wrapped,
    plot_wrapped,
    scatter_wrapped,
    set_wrap,
    stairs_wrapped,
    step_wrapped,
    vlines_wrapped,
    wrap_axes,
)
from mpl_wrap.plot import _WindowEdgeLocator

ccrs = pytest.importorskip("cartopy.crs")
from cartopy.mpl.geoaxes import GeoAxes, InterProjectionTransform

from mpl_wrap import geo

# The projections behave differently enough under set_extent and polygon
# projection that the matrix is where the real bugs turn up.
PROJECTIONS = [
    ccrs.PlateCarree(),
    ccrs.Robinson(),
    ccrs.Mercator(),
    ccrs.Orthographic(20, 30),
    ccrs.NorthPolarStereo(),
]


@pytest.fixture(params=PROJECTIONS, ids=lambda p: type(p).__name__)
def geoax(request):
    """A global GeoAxes, one per projection."""
    _, ax = plt.subplots(subplot_kw={"projection": request.param})
    ax.set_global()
    return ax


def track(n: int = 400):
    """A ground track sweeping several times round the globe, latitude in range."""
    t = np.linspace(0.0, 3.0, n)
    return -170.0 + 400.0 * t, 60.0 * np.sin(2.0 * np.pi * t)


def has_projection(transform) -> bool:
    """Whether a transform tree projects through a source CRS."""
    if isinstance(transform, InterProjectionTransform):
        return True
    return any(
        child is not None and has_projection(child)
        for child in (getattr(transform, "_a", None), getattr(transform, "_b", None))
    )


def facecolor_pixels(fig, artist) -> int:
    """Count pixels drawn in an artist's facecolor."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    want = np.round(255 * np.asarray(artist.get_facecolor()).ravel()[:3]).astype(int)
    return int((np.abs(rgba[..., :3].astype(int) - want) <= 2).all(axis=-1).sum())


# detection


def test_plain_and_polar_axes_are_not_geographic() -> None:
    _, ax = plt.subplots()
    assert not geo.is_geoaxes(ax) and not geo.geographic(ax)
    _, polar = plt.subplots(subplot_kw={"projection": "polar"})
    assert not geo.is_geoaxes(polar) and not geo.geographic(polar)


def test_geoaxes_is_geographic_by_default(geoax) -> None:
    assert geo.is_geoaxes(geoax) and geo.geographic(geoax)


def test_upgraded_geoaxes_is_still_recognised() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    wrap_axes(ax)
    # wrap_axes swaps in a derived class, which must keep GeoAxes in the MRO.
    assert type(ax).__name__ == "AxesWrapGeoAxes"
    assert isinstance(ax, GeoAxes)
    assert geo.is_geoaxes(ax) and geo.geographic(ax)


def test_geographic_flag_overrides_both_ways() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    set_wrap(ax, geographic=False)
    assert not geo.geographic(ax)
    set_wrap(ax, geographic=True)
    assert geo.geographic(ax)
    _, plain = plt.subplots()
    set_wrap(plain, geographic=True)
    assert geo.geographic(plain) and geo.crs_for(plain) is None  # no CRS off a GeoAxes


# set_wrap


def test_set_wrap_full_window_goes_global(geoax) -> None:
    before = geoax.get_xlim()
    set_wrap(geoax, wrapx=(-180.0, 180.0))
    # set_extent is degenerate over the whole globe, so set_global stands in.
    # Either way the limits stay the projection's, never the window's degrees.
    assert geoax.get_xlim() == pytest.approx(before)


def test_set_wrap_partial_window_sets_that_extent() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    set_wrap(ax, wrapx=(0.0, 180.0), wrapy=(-60.0, 60.0))
    assert ax.get_extent(crs=ccrs.PlateCarree()) == pytest.approx((0.0, 180.0, -60.0, 60.0), abs=1)


def test_set_wrap_installs_no_edge_ticks(geoax) -> None:
    set_wrap(geoax, wrapx=(-180.0, 180.0))
    assert not isinstance(geoax.xaxis.get_major_locator(), _WindowEdgeLocator)


def test_set_wrap_seam_lines_go_through_the_crs() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    set_wrap(ax, wrapx=(-180.0, 180.0), seam_lines=True)
    seams = [line for line in ax.lines if has_projection(line.get_transform())]
    assert len(seams) == 2


def test_set_wrap_warns_on_a_short_longitude_window() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    with pytest.warns(UserWarning, match="360"):
        plot_wrapped(ax, *track(), wrapx=(0.0, 180.0))


# the regression this feature exists for


def test_fill_between_renders_on_every_projection(geoax) -> None:
    lon, lat = track()
    band = fill_between_wrapped(geoax, lon, lat - 8.0, lat + 8.0, color="red", alpha=1.0)
    # Before the geographic clip fix this drew zero pixels: the clip ring was
    # built in degrees and applied through a transform in metres.
    assert facecolor_pixels(geoax.figure, band) > 500


def test_band_is_clipped_to_the_projection_boundary(geoax) -> None:
    lon, lat = track()
    band = fill_between_wrapped(geoax, lon, lat - 8.0, lat + 8.0)
    geoax.figure.canvas.draw()
    clip = band.get_clip_path().get_fully_transformed_path().get_extents()
    assert np.all(np.isfinite(clip.get_points()))
    assert clip.get_points() == pytest.approx(geoax.bbox.get_points(), abs=1)


# transform injection


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda ax: plot_wrapped(ax, *track())[0], id="plot"),
        # A PathCollection carries the data transform on its offset transform.
        pytest.param(lambda ax: scatter_wrapped(ax, *track()).get_offset_transform(), id="scatter"),
        pytest.param(lambda ax: step_wrapped(ax, *track())[0], id="step"),
        pytest.param(lambda ax: hlines_wrapped(ax, [0.0], [-100.0], [100.0]), id="hlines"),
        pytest.param(lambda ax: vlines_wrapped(ax, [0.0], [-50.0], [50.0]), id="vlines"),
        pytest.param(
            lambda ax: fill_between_wrapped(ax, track()[0], track()[1] - 5, track()[1] + 5),
            id="fill_between",
        ),
        pytest.param(
            lambda ax: stairs_wrapped(ax, np.linspace(-60, 60, 20), np.linspace(-180, 180, 21)),
            id="stairs",
        ),
    ],
)
def test_artists_get_a_projecting_transform(call) -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    result = call(ax)
    # Without an injected CRS, cartopy defaults transform to the axes projection
    # and the degrees would be read as metres.
    transform = result if hasattr(result, "transform_path") else result.get_transform()
    assert has_projection(transform)


def test_errorbar_parts_all_get_the_transform() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    lon, lat = track(40)
    container = errorbar_wrapped(ax, lon, lat, yerr=5.0, xerr=5.0, capsize=3.0, fmt="o")
    for part in (*container.lines[1], *container.lines[2]):
        # The caps and bars share the caller's transform, so they land in degrees.
        assert has_projection(part.get_transform())


# pole folding through the axes


def test_plot_folds_over_the_pole(geoax) -> None:
    lat = np.linspace(0.0, 200.0, 200)
    lon = np.full_like(lat, 30.0)
    plot_wrapped(geoax, lon, lat, color="red")
    geoax.figure.canvas.draw()  # would raise or drop points if latitudes stayed > 90


def test_scatter_folds_points_over_the_pole() -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    pts = scatter_wrapped(ax, [10.0], [100.0])
    assert list(pts.get_offsets()[0]) == pytest.approx([-170.0, 80.0])


# spans


@pytest.mark.parametrize("span", [axhspan_wrapped, axvspan_wrapped])
def test_spans_raise_on_a_geographic_axes(span) -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    with pytest.raises(NotImplementedError, match="geographic axes"):
        span(ax, 10.0, 20.0)


@pytest.mark.parametrize("span", [axhspan_wrapped, axvspan_wrapped])
def test_spans_work_once_geographic_is_turned_off(span) -> None:
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    set_wrap(ax, geographic=False)
    assert len(span(ax, 10.0, 20.0)) == 1


# pickling


def test_upgraded_geoaxes_round_trips_through_pickle() -> None:
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    wrap_axes(ax, wrapx=(-180.0, 180.0), geographic=True)
    restored = pickle.loads(pickle.dumps(fig))
    assert getattr(restored.axes[0], geo.GEO_ATTR) is True


# fill_around


def polar_track(n: int = 900):
    """A polar ground track: latitude runs past both poles, longitude past the seam."""
    u = np.linspace(0.0, 1080.0, n)
    return -180.0 + 0.4 * (u - 450.0), u


def corridor_of(lon, lat, width=2.696, limits=(geo.LAT_MIN, geo.LAT_MAX)):
    """The corridor geometry `fill_around` draws, without an axes in the way."""
    return geo.corridor_region(lon, lat, width, limits)


def test_fill_around_needs_a_geographic_axes() -> None:
    _, ax = plt.subplots()
    with pytest.raises(TypeError, match="geographic"):
        fill_around(ax, [0.0, 10.0], [0.0, 10.0], 2.696)


def test_fill_around_on_a_plain_geographic_axes() -> None:
    """Off a GeoAxes the corridor lands in the longitude window, with data limits."""
    _, ax = plt.subplots()
    set_wrap(ax, wrapx=(0.0, 360.0), geographic=True)
    artist = fill_around(ax, *polar_track(), 2.696)
    verts = np.vstack([p.vertices for p in artist.get_paths()])
    assert verts[:, 0].min() >= 0.0 and verts[:, 0].max() <= 360.0
    assert verts[:, 0].max() > 180.0  # not just the western half
    assert np.isfinite(ax.dataLim.get_points()).all()


def test_corridor_width_is_degrees_of_arc() -> None:
    """Along the equator the corridor's edges sit exactly ``width`` degrees off."""
    region = corridor_of([0.0, 10.0, 20.0], [0.0, 0.0, 0.0], width=5.0)
    west, south, east, north = region.bounds
    assert np.allclose([south, north], [-5.0, 5.0])
    assert np.allclose([west, east], [0.0, 20.0])


def test_fill_around_draws_a_corridor(geoax) -> None:
    lon, lat = polar_track()
    artist = fill_around(geoax, lon, lat, 2.696, facecolor="red")
    fig = geoax.figure
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    drawn = ((rgba[..., 0] > 200) & (rgba[..., 1] < 80) & (rgba[..., 2] < 80)).sum()
    total = rgba.shape[0] * rgba.shape[1]
    assert artist is not None
    # Enough to be visible, but nowhere near the flood that a polygon projected
    # inside out produces - which is the failure this construction avoids.
    assert 0.002 < drawn / total < 0.30


def test_fill_around_covers_the_pole_it_runs_over() -> None:
    """A track over a pole covers the whole cap within its width of it."""
    from shapely.geometry import Point

    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    region = corridor_of(*polar_track())
    # The track passes exactly through the pole, so every point within its width
    # of the pole is within its width of the track - the corridor has no hole
    # there, and no gap between the runs either side.
    assert region.contains(Point(0.0, 89.99))
    assert region.contains(Point(150.0, 89.99))  # any meridian, not just the track's
    # Still a corridor rather than a blanket. The bound is generous because a
    # lon/lat area counts the polar regions far more heavily than the globe does.
    assert region.area < 0.5 * 360.0 * 180.0


def test_fill_around_stays_inside_the_lon_lat_domain() -> None:
    """Cells straddling the antimeridian are cut, not left to wrap the globe."""
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.set_global()
    region = corridor_of(*polar_track())
    bounds = np.array([region.bounds])
    assert bounds[:, 0].min() >= -180.0 and bounds[:, 2].max() <= 180.0


# the same ground covered in every projection


def _draw_band(ax) -> None:
    lon, lat = polar_track()
    fill_between_wrapped(ax, lon, lat - 12.0, lat + 12.0, facecolor="red", edgecolor="none")


def _draw_corridor(ax) -> None:
    fill_around(ax, *polar_track(), 2.696, facecolor="red")


def probe_coverage(proj, draw, probes):
    """Which probe points a drawing covers, and which the projection can show.

    Renders on one projection, then samples the result back at fixed
    longitude/latitude points, so two projections can be compared on the ground
    rather than on the page.
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80, subplot_kw={"projection": proj})
    ax.set_global()
    ax.set_position([0, 0, 1, 1])
    set_wrap(ax)
    draw(ax)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())
    height, width = image.shape[:2]
    painted = (image[..., 0] > 180) & (image[..., 1] < 100) & (image[..., 2] < 100)

    xy = proj.transform_points(ccrs.PlateCarree(), probes[:, 0], probes[:, 1])[:, :2]
    display = ax.transData.transform(xy)
    with np.errstate(invalid="ignore"):
        column = np.nan_to_num(display[:, 0]).round().astype(int)
        row = height - 1 - np.nan_to_num(display[:, 1]).round().astype(int)
    shown = (
        np.isfinite(xy).all(axis=1) & (column >= 0) & (column < width) & (row >= 0) & (row < height)
    )
    covered = np.zeros(len(probes), dtype=bool)
    covered[shown] = painted[row[shown], column[shown]]
    plt.close(fig)
    return shown, covered


@pytest.fixture(scope="module")
def probes():
    """Longitude/latitude points to compare projections at, keeping off the poles."""
    lon, lat = np.meshgrid(np.arange(-177.0, 180.0, 3.0), np.arange(-87.0, 88.0, 3.0))
    return np.column_stack([lon.ravel(), lat.ravel()])


@pytest.mark.parametrize(
    "proj",
    [
        pytest.param(ccrs.Mollweide(), id="Mollweide"),
        pytest.param(ccrs.Robinson(), id="Robinson"),
        pytest.param(ccrs.Mercator(), id="Mercator"),
        # An azimuthal projection sees only a hemisphere, and a region that
        # surrounds a pole must not come out inside out on it.
        pytest.param(ccrs.Orthographic(-40, 35), id="Orthographic"),
        pytest.param(ccrs.NorthPolarStereo(), id="NorthPolarStereo"),
        pytest.param(ccrs.SouthPolarStereo(), id="SouthPolarStereo"),
        pytest.param(ccrs.Geostationary(), id="Geostationary"),
        pytest.param(ccrs.InterruptedGoodeHomolosine(), id="IGH"),
    ],
)
@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(_draw_band, id="fill_between_wrapped"),
        pytest.param(_draw_corridor, id="fill_around"),
    ],
)
def test_same_ground_covered_in_every_projection(proj, draw, probes) -> None:
    """A filled area is a region of the globe: every projection must show that region.

    The band is built as a closed shape and mapped onto the sphere for exactly
    this reason - folding its two edges separately gives a different area in
    each projection.
    """
    shown_flat, covered_flat = probe_coverage(ccrs.PlateCarree(), draw, probes)
    shown, covered = probe_coverage(proj, draw, probes)
    both = shown & shown_flat
    agreement = (covered[both] == covered_flat[both]).mean()
    # Short of exact: the probe grid, antialiasing and a projection's own
    # resampling all blur the boundary of a band a few degrees wide.
    assert agreement > 0.93, f"only {agreement:.1%} of the globe agrees with PlateCarree"


# ground truth, independent of any projection


@pytest.mark.parametrize("n", [2, 3, 5, 40])
def test_corridor_covers_its_own_track(n) -> None:
    """Every point of the track lies inside the corridor drawn around it.

    Checked against the track itself rather than against another projection's
    render: a mistake made the same way everywhere passes a comparison between
    projections, but cannot pass this.
    """
    from shapely.geometry import Point

    lon, lat = polar_track(n)
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    region = corridor_of(lon, lat)

    folded_lon, folded_lat = geo.fold_poles(lon, lat)
    folded_lon = (folded_lon + 180.0) % 360.0 - 180.0
    missed = [
        (round(x, 2), round(y, 2))
        for x, y in zip(folded_lon, folded_lat)
        if not region.buffer(1e-9).contains(Point(x, y))
    ]
    assert not missed, f"{len(missed)} track points fall outside their own corridor: {missed[:3]}"


def test_corridor_reaches_the_full_length_of_the_track() -> None:
    """The corridor spans the whole track, not all but the last point."""
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    lon = np.linspace(0.0, 40.0, 5)
    region = corridor_of(lon, np.zeros(5))
    assert region.bounds[0] == pytest.approx(0.0, abs=0.1)
    assert region.bounds[2] == pytest.approx(40.0, abs=0.1)


@pytest.mark.parametrize(
    "lon, lat",
    [
        pytest.param(np.array([]), np.array([]), id="empty"),
        pytest.param(np.array([0.0]), np.array([0.0]), id="one-point"),
        pytest.param(np.full(5, np.nan), np.full(5, np.nan), id="all-nan"),
        pytest.param(np.zeros(4), np.array([0.0, np.nan, np.inf, 10.0]), id="non-finite"),
    ],
)
def test_corridor_draws_nothing_from_nothing(lon, lat) -> None:
    """Degenerate input draws no corridor, rather than raising or inventing one."""
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    geoms = [corridor_of(lon, lat)]
    # A NaN is not a pole crossing, so it earns no polar cap.
    assert sum(g.area for g in geoms) == 0.0


def test_band_edge_exactly_at_a_pole_is_drawable(geoax) -> None:
    """A band whose edge touches +/-90 exactly draws, rather than raising.

    The cutting and unioning that maps a band onto the sphere can leave a stray
    point or line behind where an edge grazes a pole. Only areas are wanted.
    """
    band = fill_between_wrapped(geoax, [0.0, 10.0, 20.0], [80.0, 80.0, 80.0], [85.0, 90.0, 85.0])
    geoax.figure.canvas.draw()
    assert band is not None


def test_horizontal_stairs_keeps_its_longitudes() -> None:
    """A horizontal staircase has its latitudes on the edges, not the values."""
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()
    values = np.linspace(-170.0, 170.0, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # clamping longitudes would warn about latitudes
        patch = stairs_wrapped(ax, values, np.linspace(-80.0, 80.0, 11), orientation="horizontal")
    assert patch.get_data()[0] == pytest.approx(values)


def test_geographic_plain_axes_still_autoscales() -> None:
    """Off a GeoAxes there is no projection, so the data limits are the data."""
    _, ax = plt.subplots()
    set_wrap(ax, geographic=True)
    fill_between_wrapped(ax, np.linspace(-170.0, 170.0, 50), np.full(50, -45.0), np.full(50, 45.0))
    assert np.isfinite(ax.dataLim.get_points()).all()
    assert ax.get_xlim()[1] > 100.0  # not left at the default (0, 1)
