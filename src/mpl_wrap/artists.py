"""The artists the filled helpers return, matching their matplotlib counterparts.

`WrapFillBetween` and `WrapStepPatch` subclass the artists ``ax.fill_between``
and ``ax.stairs`` return, so the helpers keep type parity (``isinstance``) and
land in the same ``Axes`` container, with the wrapped geometry swapped in for
the straight one. Setting new data on either re-wraps it.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.patches import StepPatch
from matplotlib.path import Path

from mpl_wrap.data import _band_extent, _band_vertices

__all__ = [
    "WrapFillBetween",
    "WrapStepPatch",
]


class WrapFillBetween(FillBetweenPolyCollection):
    """The ``ax.fill_between`` collection, holding a wrapped band.

    The band is tiled at every period offset and held as one compound path, so
    the union of the tiles fills once with no double alpha, and the artist is
    clipped to the window when it is added to the axes. Wrapping is baked in:
    ``set_data`` re-wraps into the same window(s), fixed at construction.
    """

    # Set by FillBetweenPolyCollection.__init__, but absent from its type stubs.
    _interpolate: bool
    _step: str | None

    def __init__(
        self,
        x: Any,
        y1: Any,
        y2: Any,
        *,
        wrapx: np.ndarray | None = None,
        wrapy: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        self._wrapx = wrapx
        self._wrapy = wrapy
        self._band_codes: np.ndarray | None = None
        super().__init__("x", x, y1, y2, **kwargs)

    def _make_verts(self, t: Any, f1: Any, f2: Any, where: Any) -> list[np.ndarray]:
        """Build the wrapped band instead of the straight one (called by set_data too)."""
        verts, codes = _band_vertices(
            np.asarray(t, dtype=float),
            np.asarray(f1, dtype=float),
            np.asarray(f2, dtype=float),
            where=where,
            interpolate=self._interpolate,
            step=self._step,
            wrapx=self._wrapx,
            wrapy=self._wrapy,
        )
        self._band_codes = codes
        # The tiles overshoot the window; only their intersection with it is drawn.
        self._bbox = _band_extent(verts, self._wrapx, self._wrapy)
        return [verts]

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
