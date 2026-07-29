"""The artists the filled helpers return, matching their matplotlib counterparts.

`WrapFillBetween` and `WrapStepPatch` subclass the artists ``ax.fill_between``
and ``ax.stairs`` return, so the helpers keep type parity (``isinstance``) and
land in the same ``Axes`` container, with the wrapped geometry swapped in for
the straight one. Setting new data on either re-wraps it.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from matplotlib.artist import allow_rasterization
from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.patches import StepPatch
from matplotlib.path import Path

from mpl_wrap.data import _band_edges, _band_extent, _band_vertices

__all__ = [
    "WrapFillBetween",
    "WrapStepPatch",
]


class WrapFillBetween(FillBetweenPolyCollection):
    """The ``ax.fill_between`` / ``ax.fill_betweenx`` collection, holding a wrapped band.

    The band is tiled at every period offset and held as one compound path, so
    the union of the tiles fills once with no double alpha, and the artist is
    clipped to the window when it is added to the axes. Wrapping is baked in:
    ``set_data`` re-wraps into the same window(s), fixed at construction.

    ``t_direction`` is the axis the band runs along, "x" for ``fill_between`` and
    "y" for ``fill_betweenx``, as in ``FillBetweenPolyCollection``.
    """

    # Set by FillBetweenPolyCollection.__init__, but absent from its type stubs.
    _interpolate: bool
    _step: str | None
    t_direction: Literal["x", "y"]
    # Set by Collection.__init__, but absent from its type stubs.
    _edgecolors: np.ndarray
    _facecolors: np.ndarray
    _paths: list[Path]
    _hatch: str | None

    def __init__(
        self,
        t_direction: Literal["x", "y"],
        t: Any,
        f1: Any,
        f2: Any,
        *,
        wrapx: np.ndarray | None = None,
        wrapy: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        self._wrapx = wrapx
        self._wrapy = wrapy
        self._band_codes: np.ndarray | None = None
        self._edge_path: Path | None = None
        super().__init__(t_direction, t, f1, f2, **kwargs)

    def _make_verts(self, t: Any, f1: Any, f2: Any, where: Any) -> list[np.ndarray]:
        """Build the wrapped band instead of the straight one (called by set_data too).

        Built with ``t`` along x and swapped for ``fill_betweenx``, so the tiling
        only ever works in one direction.
        """
        along_x = self.t_direction == "x"
        wrap_t, wrap_f = (self._wrapx, self._wrapy) if along_x else (self._wrapy, self._wrapx)
        band = {
            "where": where,
            "interpolate": self._interpolate,
            "step": self._step,
            "wrapx": wrap_t,
            "wrapy": wrap_f,
        }
        t, f1, f2 = (np.asarray(a, dtype=float) for a in (t, f1, f2))
        verts, codes = _band_vertices(t, f1, f2, **band)
        edge_verts, edge_codes = _band_edges(t, f1, f2, **band)
        if not along_x:
            verts = verts[:, ::-1]
            edge_verts = edge_verts[:, ::-1]
        self._band_codes = codes
        self._edge_path = Path(edge_verts, edge_codes) if len(edge_verts) else None
        # The tiles overshoot the window. Only their intersection is drawn.
        self._bbox = _band_extent(verts, self._wrapx, self._wrapy)
        return [verts]

    @allow_rasterization
    def draw(self, renderer: Any) -> None:
        """Fill the tiles with no edge, then stroke the band edge path.

        The compound fill path has subpath boundaries interior to the union -
        tile joins, and the sides of a saturated run's rectangle - so stroking
        it would draw seams across the band. The edge, when visible, is instead
        stroked from the wrapped band boundary built by ``_make_verts``.
        """
        # Resolve colours now: the first update_scalarmappable call inside
        # Collection.draw resets them from the originals, undoing the swaps below.
        self.update_scalarmappable()
        edge = np.asarray(self.get_edgecolor(), dtype=float)
        widths = np.asarray(self.get_linewidth(), dtype=float)
        if (
            self._edge_path is None
            or self.get_array() is not None  # mapped colours repopulate during draw
            or edge.size == 0
            or not edge[:, 3].any()
            or not widths.any()
        ):
            super().draw(renderer)
            return
        state = self._paths, self._facecolors, self._edgecolors, self._hatch
        try:
            self._edgecolors = np.empty((0, 4))
            super().draw(renderer)
            self._paths = [self._edge_path]
            self._facecolors = np.empty((0, 4))
            self._edgecolors = edge  # resolved, in case it was "face"
            self._hatch = None
            super().draw(renderer)
        finally:
            self._paths, self._facecolors, self._edgecolors, self._hatch = state

    def set_verts(self, verts: Any, closed: bool = True) -> None:
        """Set the tiles as one compound path, so the union fills once."""
        if self._band_codes is None:
            super().set_verts(verts, closed)
        else:
            self.set_verts_and_codes(verts, [self._band_codes])  # type: ignore[list-item]


class WrapStepPatch(StepPatch):
    """The ``ax.stairs`` patch, drawing a wrapped staircase.

    ``values``, ``edges`` and ``baseline`` are kept exactly as passed - so
    ``get_data`` round-trips and ``set_data`` re-wraps - while the drawn path is
    the wrapped staircase (or, filled, the tiled band).
    """

    # Set by StepPatch.__init__, but absent from its type stubs.
    _values: np.ndarray
    _edges: np.ndarray
    _baseline: np.ndarray | None

    def __init__(
        self,
        values: Any,
        edges: Any,
        *,
        build: Callable[[np.ndarray, np.ndarray, np.ndarray | None, str], Path],
        **kwargs: Any,
    ) -> None:
        self._build = build
        super().__init__(values, edges, **kwargs)

    def _update_path(self) -> None:
        """Wrap the staircase instead of drawing it straight (called by set_data too)."""
        self._path = self._build(self._values, self._edges, self._baseline, self.orientation)
