"""Geographic coordinates: longitude/latitude in degrees, and cartopy axes.

Three layers, all internal to mpl_wrap.

Pole folding. Latitude is not periodic: a track running past the north pole
comes back down the antipodal meridian, 180 degrees away in longitude.
`fold_poles` folds points and `pole_polyline` folds a polyline, routing the
line to the pole and breaking it there. Longitude is periodic and is wrapped by
`mpl_wrap.data` as usual.

Regions. A filled band on the globe is built as a closed shape in the
(longitude, unwrapped latitude) plane and mapped onto the sphere by
`to_sphere`, so it covers the same ground in every projection. This needs
shapely.

Axes. A cartopy ``GeoAxes`` has ``transData`` in projected units (metres, for
most projections) while the data is in degrees, carried by a ``transform=``
CRS, so limits, clip paths and seam lines go through that CRS. cartopy is
optional: it is imported lazily and only for an axes that is already a
``GeoAxes``, so importing this module never loads it.
"""

import sys
from typing import Any, NamedTuple

import numpy as np
from matplotlib.axes import Axes

__all__ = [
    "unfold_poles",
]

# Latitude is bounded by the poles and is not periodic: a track running past a
# pole comes back down the far side, at the antipodal longitude.
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0
_LAT_PERIOD = LAT_MAX - LAT_MIN  # 180: the latitude span from pole to pole
_ANTIPODE = 180.0  # the longitude shift applied on crossing a pole

# Regions are densified so that no edge is longer than this, in degrees. A
# projection bends each straight lon/lat edge into a curve, and given only the
# endpoints of a long edge it can put the inside of the polygon on the wrong
# side. Near a pole a few degrees of longitude is most of the way round the map.
_MAX_EDGE = 1.0


def _shapely() -> Any:
    """The shapely package, which builds the regions, with a clear error if missing."""
    try:
        import shapely  # type: ignore[import-untyped]
        import shapely.affinity  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Filled regions on a geographic axes need shapely. Install it with "
            "'pip install mpl_wrap[geo]' (cartopy already includes it)."
        ) from exc
    return shapely


def _ccrs() -> Any:
    """The ``cartopy.crs`` module, with a clear error if cartopy is missing.

    Only reached for an axes that is already a ``GeoAxes``, so in practice
    cartopy is already imported by the time this runs.
    """
    try:
        import cartopy.crs as ccrs  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "A GeoAxes needs cartopy. Install it with 'pip install cartopy'."
        ) from exc
    return ccrs


def pole_bands(lat: np.ndarray) -> np.ndarray:
    """The pole-to-pole band each latitude falls in.

    Band ``m`` spans ``[-90 + 180m, 90 + 180m)``, so band 0 is the real
    latitude range and an out-of-range latitude sits in a band either side.
    """
    return np.floor((lat - LAT_MIN) / _LAT_PERIOD)


def fold_poles(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold out-of-range latitudes back over the pole, pointwise.

    A latitude past a pole is reflected back over it and its longitude moved to
    the antipodal meridian, which is where that point is on the globe:
    ``lat=100`` is 80 degrees north at ``lon+180``. Odd bands are the reflected
    ones. In-range and non-finite latitudes pass through with their longitude
    untouched.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    finite = np.isfinite(lat)
    m = pole_bands(np.where(finite, lat, 0.0))
    resid = np.where(finite, lat - _LAT_PERIOD * m, lat)
    reflected = finite & ((m % 2) != 0)
    return (
        np.where(reflected, lon + _ANTIPODE, lon),
        np.where(reflected, -resid, resid),
    )


def unfold_poles(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """Make a folded longitude/latitude track continuous past the poles.

    Geodetic routines keep latitude within the poles, so a track passing over
    a pole arrives as a jump of 180 degrees in longitude at the pole. This is
    the inverse of `fold_poles`: at each such pass the latitude continues past
    the pole and the longitude is shifted back to the meridian the track was
    on, so the result can be passed straight to `plot_wrapped` on a geographic
    axes. Longitude is also unwrapped across the antimeridian.

    A pass is a step of more than 90 degrees in longitude between consecutive
    samples, at the pole given by the sign of the latitude there. That takes
    the track to be sampled finely enough that consecutive samples are within
    a quarter turn of each other in longitude, as a ground track is. Passes
    over alternating poles carry the track on past each one, and two passes
    over the same pole turn it back.

    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude in degrees, latitude within +/-90.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The continuous longitude and latitude. Non-finite samples pass through
        and do not count as a pass.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if len(lon) == 0:
        return lon.copy(), lat.copy()
    finite = np.isfinite(lon) & np.isfinite(lat)
    lon_ok, lat_ok = np.where(finite, lon, 0.0), np.where(finite, lat, 0.0)
    step = (np.diff(lon_ok) + _ANTIPODE) % (2 * _ANTIPODE) - _ANTIPODE
    passes = (np.abs(step) > 0.5 * _ANTIPODE) & finite[:-1] & finite[1:]
    # From an even band the north pole leads up and the south pole down, and
    # from an odd band the reverse, so alternating poles carry the track on
    # past each one and the same pole twice turns it back.
    at = np.nonzero(passes)[0]
    north = lat_ok[at] + lat_ok[at + 1] > 0
    even = np.arange(len(at)) % 2 == 0
    steps = np.zeros(len(lon) - 1)
    steps[at] = np.where(north == even, 1.0, -1.0)
    band = np.concatenate([[0.0], np.cumsum(steps)])
    odd = band % 2 != 0
    out_lat = np.where(odd, _LAT_PERIOD * band - lat, lat + _LAT_PERIOD * band)
    out_lon = np.where(odd, lon - _ANTIPODE, lon)
    # With the passes undone, what is left is ordinary antimeridian wrapping.
    step = (np.diff(np.where(finite, out_lon, 0.0)) + _ANTIPODE) % (2 * _ANTIPODE) - _ANTIPODE
    unwrapped = out_lon[0] + np.concatenate([[0.0], np.cumsum(step)])
    return np.where(finite, unwrapped, lon), np.where(finite, out_lat, lat)


def pole_polyline(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold a polyline's latitude over the poles as one NaN-broken polyline.

    Vectorized like `_wrap_polyline`. Each band boundary a segment crosses is a
    pole, so the line is routed up to the pole, broken with a NaN, and resumed
    from the same pole at the antipodal longitude. Longitude is left unwrapped
    for `_wrap_polyline` to fold into the longitude window afterwards.

    Returns the folded coordinates plus the output index of each input sample,
    since the routing inserts vertices in between.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lat)
    if n == 0:
        return lon.copy(), lat.copy(), np.empty(0, dtype=np.intp)

    finite = np.isfinite(lon) & np.isfinite(lat)
    m = np.where(finite, pole_bands(np.where(finite, lat, 0.0)), 0.0)
    fold_lon, fold_lat = fold_poles(lon, lat)

    dm = np.where(finite[:-1] & finite[1:], m[1:] - m[:-1], 0.0).astype(np.int64)
    ncross = np.abs(dm)
    total = int(ncross.sum())
    if total == 0:
        return fold_lon, fold_lat, np.arange(n)

    before = np.concatenate([[0], np.cumsum(ncross)])  # crossings before each sample
    sample_idx = np.arange(n) + 3 * before
    out_lon = np.empty(n + 3 * total)
    out_lat = np.empty_like(out_lon)
    out_lon[sample_idx] = fold_lon
    out_lat[sample_idx] = fold_lat

    seg = np.repeat(np.arange(n - 1), ncross)  # segment each crossing belongs to
    rank = np.arange(total) - np.repeat(before[:-1], ncross)  # rank within segment
    asc = dm[seg] > 0
    level = np.where(asc, m[:-1][seg] + 1 + rank, m[:-1][seg] - rank)
    lat_c = LAT_MIN + level * _LAT_PERIOD  # the unfolded latitude of the boundary
    lat_i, lat_j = lat[:-1][seg], lat[1:][seg]
    lon_i, lon_j = lon[:-1][seg], lon[1:][seg]
    lon_c = lon_i + (lat_c - lat_i) / (lat_j - lat_i) * (lon_j - lon_i)

    # Odd boundaries are the north pole, even ones the south. The bands either
    # side of it decide whether each end of the route is on the antipodal meridian.
    pole = np.where(level % 2 != 0, LAT_MAX, LAT_MIN)
    exit_band = np.where(asc, level - 1, level)
    enter_band = np.where(asc, level, level - 1)

    start = sample_idx[:-1][seg] + 1 + 3 * rank  # first of the crossing's 3 vertices
    out_lat[start] = out_lat[start + 2] = pole
    out_lat[start + 1] = np.nan
    out_lon[start] = lon_c + _ANTIPODE * (exit_band % 2)
    out_lon[start + 1] = np.nan
    out_lon[start + 2] = lon_c + _ANTIPODE * (enter_band % 2)
    return out_lon, out_lat, sample_idx


def _polygons(rings: Any) -> Any:
    """Union rings into one valid area, dropping anything that is not a polygon.

    Self-crossing rings resolve into valid parts under ``buffer(0)``. Cutting
    and unioning can leave stray points and lines behind, such as a band edge
    that just touches a pole, and only areas are kept.
    """
    shapely = _shapely()
    areas: list[Any] = []
    for ring in rings:
        shape = ring if ring.is_valid else ring.buffer(0)
        areas.extend(
            p for p in shapely.get_parts(shape) if p.geom_type == "Polygon" and not p.is_empty
        )
    if not areas:
        return shapely.Polygon()
    return shapely.unary_union(shapely.MultiPolygon(areas) if len(areas) > 1 else areas[0])


def _runs(good: np.ndarray) -> Any:
    """The index runs of consecutive usable samples, each long enough to fill."""
    for run in np.split(np.arange(len(good)), np.nonzero(~good)[0]):
        run = run[good[run]]
        if len(run) >= 2:
            yield run


def band_region(
    t: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    along_lon: bool,
    limits: tuple[float, float],
    lon_min: float = LON_MIN,
) -> Any:
    """The closed region a band covers on the globe, as a shapely geometry.

    The ring runs out along one edge and back along the other in the
    (longitude, unwrapped latitude) plane, where it is an ordinary polygon, and
    `to_sphere` maps it onto the globe. ``limits`` and ``lon_min`` are passed
    through to it.
    """
    shapely = _shapely()
    t, f1, f2 = (np.asarray(v, dtype=float) for v in (t, f1, f2))
    good = np.isfinite(t) & np.isfinite(f1) & np.isfinite(f2)

    rings = []
    for run in _runs(good):
        # Out along one edge and back along the other. For fill_betweenx the two
        # edges are longitudes and ``t`` is the latitude they share.
        if along_lon:
            ring = np.vstack([np.c_[t[run], f1[run]], np.c_[t[run][::-1], f2[run][::-1]]])
        else:
            ring = np.vstack([np.c_[f1[run], t[run]], np.c_[f2[run][::-1], t[run][::-1]]])
        rings.append(shapely.Polygon(ring))
    if not rings:
        return shapely.Polygon()
    return to_sphere(_polygons(rings), limits, lon_min)


def to_sphere(
    shape: Any,
    limits: tuple[float, float] = (LAT_MIN, LAT_MAX),
    lon_min: float = LON_MIN,
) -> Any:
    """Map a polygon drawn in the unwrapped lon/lat plane onto the sphere.

    That plane covers the sphere many times over. The shape is cut along every
    pole line and every antimeridian, each piece is moved to where it lies on
    the globe (reflected over the pole and onto the antipodal meridian for an
    odd band of latitude, translated a whole turn in longitude), and the pieces
    are unioned back together. ``limits`` trims the result to the latitudes the
    target projection can place a vertex at, and ``lon_min`` is the west edge
    of the longitude window the result lands in.

    The result is densified to edges of at most `_MAX_EDGE` degrees, which
    leaves the region and its area unchanged.
    """
    shapely = _shapely()
    affine_transform = shapely.affinity.affine_transform
    if shape.is_empty:
        return shapely.Polygon()

    def unfold_pole(piece: Any, m: int) -> Any:
        # An odd band is reflected over the pole and onto the antipodal meridian.
        if m % 2:
            return affine_transform(piece, [1, 0, 0, -1, _ANTIPODE, _LAT_PERIOD * m])
        return affine_transform(piece, [1, 0, 0, 1, 0, -_LAT_PERIOD * m])

    def unturn(piece: Any, k: int) -> Any:
        return affine_transform(piece, [1, 0, 0, 1, -2 * _ANTIPODE * k, 0])

    sphere = _tiles(shape, True, LAT_MIN, _LAT_PERIOD, unfold_pole)
    if sphere.is_empty:
        return shapely.Polygon()
    world = _tiles(sphere, False, lon_min, 2 * _ANTIPODE, unturn)
    if world.is_empty:
        return shapely.Polygon()
    window = shapely.box(lon_min, limits[0], lon_min + 2 * _ANTIPODE, limits[1])
    return world.intersection(window).segmentize(_MAX_EDGE)


def _tiles(shape: Any, vertical: bool, origin: float, period: float, move: Any) -> Any:
    """Cut a shape into period-wide strips along one axis and move each into range.

    ``move(piece, k)`` returns strip ``k`` placed in the base period, where
    strip 0 starts at ``origin``.
    """
    box = _shapely().box
    west, south, east, north = shape.bounds
    low, high = (south, north) if vertical else (west, east)
    pieces = []
    first, last = (int(np.floor((v - origin) / period)) for v in (low, high))
    for k in range(first, last + 1):
        lo, hi = origin + period * k, origin + period * (k + 1)
        strip = box(west, lo, east, hi) if vertical else box(lo, south, hi, north)
        piece = shape.intersection(strip)
        if not piece.is_empty:
            pieces.append(move(piece, k))
    return _polygons(pieces)


def to_window(shape: Any, wrapx: Any, wrapy: Any) -> Any:
    """Fold a polygon drawn in the unwrapped plane into the wrap window(s).

    Along each wrapped axis the shape is cut at every period boundary and the
    pieces are translated into the window and unioned. An unwrapped axis
    (window None) is left alone.
    """
    translate = _shapely().affinity.translate
    for vertical, window in ((False, wrapx), (True, wrapy)):
        if window is None or shape.is_empty:
            continue
        lo, hi = (float(v) for v in window)
        period = hi - lo

        def move(piece: Any, k: int, vertical: bool = vertical, period: float = period) -> Any:
            return (
                translate(piece, yoff=-period * k)
                if vertical
                else translate(piece, xoff=-period * k)
            )

        shape = _tiles(shape, vertical, lo, period, move)
    return shape


def plane_corridor_region(
    x: np.ndarray, y: np.ndarray, width: Any, wrapx: Any = None, wrapy: Any = None
) -> Any:
    """The closed region a corridor of constant width covers in the data plane.

    ``width`` is the half-width in data units, offset perpendicular to the
    track, which takes x and y to share a unit. Each leg contributes a
    rectangle and each interior vertex a round join, the ends are flat, and the
    corridor is broken at non-finite samples. The result is folded into the
    wrap window(s) by `to_window`.
    """
    shapely = _shapely()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:  # no pair of points, so no direction to offset from
        return shapely.Polygon()
    half = np.broadcast_to(np.asarray(width, dtype=float), x.shape)
    finite = np.isfinite(x) & np.isfinite(y)

    dx, dy = np.diff(np.where(finite, x, 0.0)), np.diff(np.where(finite, y, 0.0))
    length = np.hypot(dx, dy)
    leg = finite[:-1] & finite[1:] & (length > 0)
    length = np.where(leg, length, 1.0)
    normal = np.column_stack([-dy / length, dx / length])
    at = np.column_stack([x, y])
    pieces = []
    for i in np.nonzero(leg)[0]:
        offset0, offset1 = half[i] * normal[i], half[i + 1] * normal[i]
        pieces.append(
            shapely.Polygon(
                [at[i] + offset0, at[i + 1] + offset1, at[i + 1] - offset1, at[i] - offset0]
            )
        )
    for i in np.nonzero(leg[:-1] & leg[1:])[0] + 1:
        pieces.append(shapely.Point(at[i]).buffer(half[i]))
    if not pieces:
        return shapely.Polygon()
    return to_window(_polygons(pieces), wrapx, wrapy)


def region_path(geometry: Any) -> tuple[np.ndarray, np.ndarray]:
    """A shapely region as matplotlib vertices and codes, one closed ring each."""
    from matplotlib.path import Path

    verts: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for polygon in _shapely().get_parts(geometry):
        if polygon.is_empty or polygon.geom_type != "Polygon":
            continue
        for ring in [polygon.exterior, *polygon.interiors]:
            xy = np.asarray(ring.coords, dtype=float)
            if len(xy) < 3:
                continue
            verts.append(xy)
            codes.append(np.array([Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]))
    if not verts:
        return np.empty((0, 2)), np.empty(0, dtype=np.uint8)
    return np.vstack(verts), np.concatenate(codes).astype(np.uint8)


def corridor_region(
    lon: np.ndarray,
    lat: np.ndarray,
    width: Any,
    limits: tuple[float, float],
    lon_min: float = LON_MIN,
) -> Any:
    """The closed region a corridor of constant width covers on the globe.

    ``width`` is the half-width in degrees of arc, offset perpendicular to the
    track on a sphere. Each point is offset along the bearing of the leg leaving
    it, which suits a smoothly sampled track such as a ground track. Latitudes
    past a pole are folded over it first, so a continuous ground track goes
    straight in. Where the track runs over a pole the corridor includes the
    polar cap within ``width`` of it.
    """
    shapely = _shapely()
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if len(lon) < 2:  # no pair of points, so no direction to offset from
        return shapely.Polygon()

    # A segment that straddles a pole has no single direction, since its two
    # ends sit on opposite meridians once folded, so the corridor is broken
    # across it. A sample is inserted either side of each pole the track
    # crosses first, which keeps that break to a hair's width of track. A
    # coarsely sampled segment can clear several poles in one step.
    bands = pole_bands(lat)
    finite = np.isfinite(lon) & np.isfinite(lat)
    gap = 1e-9  # small enough to be invisible, large enough to land in its band
    out_lon, out_lat = list(lon), list(lat)
    added = 0
    for i in np.nonzero((np.diff(bands) != 0) & finite[:-1] & finite[1:])[0]:
        span = lat[i + 1] - lat[i]
        rising = span > 0
        levels = (
            range(int(bands[i]) + 1, int(bands[i + 1]) + 1)
            if rising
            else range(int(bands[i]), int(bands[i + 1]), -1)
        )
        for level in levels:
            pole = LAT_MIN + _LAT_PERIOD * level
            crossing_lon = lon[i] + (pole - lat[i]) / span * (lon[i + 1] - lon[i])
            at = i + 1 + added
            out_lon[at:at] = [crossing_lon, crossing_lon]
            out_lat[at:at] = [pole - gap, pole + gap] if rising else [pole + gap, pole - gap]
            added += 2
    lon, lat = np.asarray(out_lon, dtype=float), np.asarray(out_lat, dtype=float)

    half = np.broadcast_to(np.asarray(width, dtype=float), lon.shape)
    bands = pole_bands(lat)  # from the unfolded latitude: folding erases the crossings
    fold_lon, fold_lat = fold_poles(lon, lat)
    points = np.column_stack([fold_lon, fold_lat])
    # The corridor is also broken at non-finite samples.
    finite = np.isfinite(points).all(axis=1)
    joined = np.append(np.diff(bands) == 0, False) & finite & np.append(finite[1:], False)

    # Spherical trigonometry, in radians: the bearing of each leg, carried to
    # the final point, then each point moved ``half`` degrees of arc along the
    # bearings 90 degrees either side of it. Broken samples are kept out of the
    # trig with placeholder coordinates.
    lon_r = np.radians(np.where(finite, fold_lon, 0.0))
    lat_r = np.radians(np.where(finite, fold_lat, 0.0))
    dlon = np.diff(lon_r)
    east = np.sin(dlon) * np.cos(lat_r[1:])
    north = np.cos(lat_r[:-1]) * np.sin(lat_r[1:]) - np.sin(lat_r[:-1]) * np.cos(
        lat_r[1:]
    ) * np.cos(dlon)
    bearing = np.arctan2(east, north)
    bearing = np.append(bearing, bearing[-1])
    dist = np.radians(half)
    sides = []
    for side in (-0.5 * np.pi, 0.5 * np.pi):
        az = bearing + side
        lat2 = np.arcsin(np.sin(lat_r) * np.cos(dist) + np.cos(lat_r) * np.sin(dist) * np.cos(az))
        lon2 = lon_r + np.arctan2(
            np.sin(az) * np.sin(dist) * np.cos(lat_r), np.cos(dist) - np.sin(lat_r) * np.sin(lat2)
        )
        sides.append(np.column_stack([np.degrees(lon2), np.degrees(lat2)]))
    left, right = sides

    rings = []
    for run in _runs(joined | np.append(False, joined[:-1])):
        # Unwrap the track's longitude, then hang each edge off its own track
        # point. The step from one edge to the other at the far end of the ring
        # is a real jump of up to twice the width, most of a turn near a pole,
        # so the ring itself is never unwrapped.
        centre = np.unwrap(points[run][:, 0], period=360.0)
        sides = []
        for edge in (left, right):
            offset = (edge[run][:, 0] - points[run][:, 0] + _ANTIPODE) % 360.0 - _ANTIPODE
            sides.append(np.column_stack([centre + offset, edge[run][:, 1]]))
        rings.append(shapely.Polygon(np.vstack([sides[0], sides[1][::-1]])))

    # The track passes exactly through each pole it runs over, so every point
    # within ``width`` of the pole is within ``width`` of the track: the polar
    # cap fills the gap between the runs either side. In the unwrapped plane
    # the cap is a rectangle spanning every longitude.
    crossings = np.diff(bands) != 0
    for crossing in np.nonzero(crossings & finite[:-1] & finite[1:])[0]:
        pole = LAT_MAX if bands[crossing + 1] > bands[crossing] else LAT_MIN
        if bands[crossing] % 2:  # a reflected band runs the other way
            pole = -pole
        cap = float(np.max(half[crossing : crossing + 2]))
        edge = pole - np.sign(pole) * cap
        rings.append(shapely.box(-_ANTIPODE, min(pole, edge), _ANTIPODE, max(pole, edge)))

    if not rings:
        return shapely.Polygon()
    return to_sphere(_polygons(rings), limits, lon_min)


# --- cartopy axes -----------------------------------------------------------

# Set on an axes by set_wrap. Absent means "decide from the axes type".
GEO_ATTR = "_mpl_wrap_geographic"


def is_geoaxes(ax: Axes) -> bool:
    """Whether an axes is a cartopy ``GeoAxes``, without importing cartopy.

    If cartopy has not been imported then nothing can be a ``GeoAxes``, so the
    check is a lookup in the module table. An axes upgraded by `wrap_axes`
    keeps ``GeoAxes`` in its MRO and is recognised here.
    """
    mod = sys.modules.get("cartopy.mpl.geoaxes")
    return mod is not None and isinstance(ax, mod.GeoAxes)


def geographic(ax: Axes) -> bool:
    """Whether to treat this axes' data as longitude/latitude in degrees.

    A cartopy ``GeoAxes`` is geographic by default, and any axes can be forced
    either way with ``set_wrap(ax, geographic=...)``.
    """
    stored = getattr(ax, GEO_ATTR, None)
    return is_geoaxes(ax) if stored is None else bool(stored)


class Geo(NamedTuple):
    """How a plotting call treats geographic coordinates.

    ``on`` is whether to fold latitude at the poles. ``crs`` is the source
    coordinate reference system the data is in, or None on a plain axes, which
    can be geographic (degrees on a rectilinear plot) with no CRS to project
    through.
    """

    on: bool
    crs: Any | None


def resolve(ax: Axes) -> Geo:
    """Geographic mode for a call with no artist, such as the pure-data helpers."""
    return Geo(geographic(ax), crs_for(ax))


def setup(ax: Axes, kwargs: dict[str, Any]) -> Geo:
    """Resolve geographic mode, defaulting the artist transform to the source CRS.

    ``kwargs`` is modified in place: on a ``GeoAxes`` a missing ``transform``
    is filled in with plain lon/lat, which is what cartopy needs to place data
    in degrees. A ``transform`` that is a projected CRS (metres) means the data
    is not in degrees, and pole folding is off for that call.
    """
    if not geographic(ax):
        return Geo(False, None)
    if not is_geoaxes(ax):
        return Geo(True, None)
    crs = kwargs.get("transform")
    if crs is None:
        crs = _ccrs().PlateCarree()
        kwargs["transform"] = crs
    try:  # a projected CRS (metres) has limits well beyond the poles
        degrees = tuple(crs.y_limits) == (LAT_MIN, LAT_MAX)
    except (AttributeError, TypeError):
        degrees = False
    return Geo(degrees, crs)


# One entry per projection: the search costs a few hundred transforms.
_pole_limits: dict[Any, tuple[float, float]] = {}


def pole_limits(ax: Axes) -> tuple[float, float]:
    """How close to each pole this axes' projection can put a vertex.

    Most projections return a non-finite coordinate at a pole and for some way
    around it, and one such vertex makes a whole polygon project inside out.
    The distance varies, and some projections never reach a pole: a
    geostationary view ends near 81 degrees, an orthographic one at its own
    horizon. The limit is found by bisection and cached per projection.

    Returns the southern and northern latitude limits, in degrees.
    """
    if not is_geoaxes(ax):
        return LAT_MIN, LAT_MAX
    projection = ax.projection  # type: ignore[attr-defined]
    try:
        cached = _pole_limits.get(projection)
    except TypeError:  # an unhashable projection: measure it every time
        cached = None
    if cached is not None:
        return cached

    plate_carree = _ccrs().PlateCarree()
    # An off-centre projection reaches furthest along its own central meridian,
    # so that meridian is sampled as well as the round-number ones.
    centre = float(projection.proj4_params.get("lon_0", 0.0))
    meridians = [*np.linspace(-_ANTIPODE, _ANTIPODE, 5), centre, centre + _ANTIPODE]

    def reaches(lat: float) -> bool:
        """Whether this latitude projects finitely anywhere along it."""
        return any(
            np.isfinite(projection.transform_point(lon, lat, plate_carree)).all()
            for lon in meridians
        )

    limits = []
    for pole in (LAT_MIN, LAT_MAX):
        if reaches(pole):
            limits.append(pole)
            continue
        near, far = 0.0, pole  # near the equator is always fine, the pole is not
        for _ in range(20):  # to well under a kilometre
            middle = 0.5 * (near + far)
            near, far = (middle, far) if reaches(middle) else (near, middle)
        limits.append(near)

    result = (limits[0], limits[1])
    try:
        _pole_limits[projection] = result
    except TypeError:
        pass
    return result


def crs_for(ax: Axes) -> Any | None:
    """The CRS to express this axes' windows in, or None if it takes plain data.

    Only a geographic ``GeoAxes`` has one. Everywhere else the wrap windows are
    in data coordinates, which is what the axis methods take.
    """
    if not (geographic(ax) and is_geoaxes(ax)):
        return None
    return _ccrs().PlateCarree()


def set_lims(ax: Axes, crs: Any, wrapx: Any, wrapy: Any) -> None:
    """Set a GeoAxes' extent from the wrap window(s), in the window's own CRS.

    The window goes through ``set_extent``, since ``set_xlim`` and ``set_ylim``
    take projected units. ``set_extent`` raises on an extent covering the whole
    globe, which is degenerate in most projections, and ``set_global`` is used
    there.
    """
    x0, x1 = wrapx if wrapx is not None else crs.x_limits
    y0, y1 = wrapy if wrapy is not None else crs.y_limits
    try:
        ax.set_extent([x0, x1, y0, y1], crs=crs)  # type: ignore[attr-defined]
    except ValueError:
        ax.set_global()  # type: ignore[attr-defined]


def seam_line(
    ax: Axes, crs: Any, value: float, vertical: bool, window: Any, style: dict[str, Any]
) -> None:
    """Draw one window-edge line on a GeoAxes, in the source CRS.

    The line is plotted through the CRS from two endpoints. cartopy densifies
    the path when it projects it, so the line follows the projection's curve.
    """
    lo, hi = window if window is not None else (crs.y_limits if vertical else crs.x_limits)
    across = [lo, hi]
    along = [value, value]
    x, y = (along, across) if vertical else (across, along)
    ax.plot(x, y, transform=crs, **style)
