"""The mpl_wrap helpers as Axes methods: the ``AxesWrap`` class.

``AxesWrap`` is a regular ``matplotlib.axes.Axes`` subclass that exposes the
mpl_wrap helpers as methods (``ax.plot_wrapped(...)``). Create one with the
``"wrap"`` projection, or upgrade an existing axes of any projection in place
with `wrap_axes`. The inherited plotting methods are untouched, so wrapped and
unwrapped artists mix freely on one axes.
"""

from typing import Any

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.projections import register_projection

from mpl_wrap import wrap as _wrap
from mpl_wrap.wrap import WrapSpec

__all__ = [
    "AxesWrap",
    "AxesWrapBase",
    "wrap_axes",
]


class AxesWrapBase(Axes):
    """Base class of every wrap-enabled axes, adding the mpl_wrap helper methods.

    Not used directly: `AxesWrap` is its rectilinear form, and `wrap_axes`
    derives a variant from it for the class of an existing axes of any
    projection. Useful as the ``isinstance`` check for "has the wrap methods".
    """

    # The original (importable) class the variant was derived from, for pickling.
    _axes_wrap_base: type[Axes] = Axes

    def set_wrap(self, *args: Any, **kwargs: Any) -> "AxesWrapBase":
        """Store wrap window(s) on this axes. See `mpl_wrap.set_wrap`. Returns self."""
        _wrap.set_wrap(self, *args, **kwargs)
        return self

    def plot_wrapped(self, *args: Any, **kwargs: Any) -> list[Line2D]:
        """Plot a line wrapped into the window. See `mpl_wrap.plot_wrapped`."""
        return _wrap.plot_wrapped(self, *args, **kwargs)

    def scatter_wrapped(self, *args: Any, **kwargs: Any) -> PathCollection:
        """Scatter points folded into the window. See `mpl_wrap.scatter_wrapped`."""
        return _wrap.scatter_wrapped(self, *args, **kwargs)

    def fill_between_wrapped(self, *args: Any, **kwargs: Any) -> PathPatch:
        """Fill a band wrapped into the window. See `mpl_wrap.fill_between_wrapped`."""
        return _wrap.fill_between_wrapped(self, *args, **kwargs)

    def stairs_wrapped(self, *args: Any, **kwargs: Any) -> list[Line2D]:
        """Step plot wrapped into the window. See `mpl_wrap.stairs_wrapped`."""
        return _wrap.stairs_wrapped(self, *args, **kwargs)

    def errorbar_wrapped(self, *args: Any, **kwargs: Any) -> ErrorbarContainer:
        """Errorbars wrapped into the window. See `mpl_wrap.errorbar_wrapped`."""
        return _wrap.errorbar_wrapped(self, *args, **kwargs)

    def __reduce__(self) -> tuple[Any, ...]:
        # Wrap classes made by _axes_wrap_class cannot be pickled by reference,
        # so pickle the importable base class and re-derive the class on load.
        return (_rebuild_axes_wrap, (self._axes_wrap_base,), self.__getstate__())


class AxesWrap(AxesWrapBase):
    """A rectilinear ``Axes`` with the mpl_wrap helpers available as methods.

    Create one with the ``"wrap"`` projection::

        fig, ax = plt.subplots(subplot_kw={"projection": "wrap"})
        ax.set_wrap(wrapy=(0, 360))
        ax.plot_wrapped(x, y)

    or upgrade an existing axes with `wrap_axes`.
    """

    name = "wrap"


register_projection(AxesWrap)


# One wrap-enabled class per base class, so repeated upgrades and unpickling
# reuse the same type.
_wrap_classes: dict[type[Axes], type[AxesWrapBase]] = {}


def _axes_wrap_class(base: type[Axes]) -> type[AxesWrapBase]:
    """Return the wrap-enabled class for a given ``Axes`` (sub)class."""
    if issubclass(base, AxesWrapBase):
        return base
    if base is Axes:
        return AxesWrap
    if base not in _wrap_classes:
        cls = type(f"AxesWrap{base.__name__}", (AxesWrapBase, base), {"_axes_wrap_base": base})
        _wrap_classes[base] = cls
    return _wrap_classes[base]


def _rebuild_axes_wrap(base: type[Axes]) -> AxesWrapBase:
    """Recreate an empty wrapped axes for unpickling (state is applied after)."""
    cls = _axes_wrap_class(base)
    return cls.__new__(cls)


def wrap_axes(
    ax: Axes,
    wrapx: WrapSpec = None,
    wrapy: WrapSpec = None,
    *,
    set_lims: bool = True,
    seam_lines: bool = False,
    margin: float = 0.05,
    seam_kwargs: dict[str, Any] | None = None,
) -> AxesWrapBase:
    """Upgrade an existing axes in place to an `AxesWrap`.

    The axes' class is swapped for a wrap-enabled variant of its current class,
    adding the ``*_wrapped`` methods while keeping its projection (rectilinear,
    polar, ...) and all existing state. Window arguments are forwarded to
    `set_wrap`. Upgraded axes remain picklable.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to upgrade, modified in place.
    wrapx, wrapy, set_lims, seam_lines, margin, seam_kwargs
        Forwarded to `set_wrap`.

    Returns
    -------
    AxesWrapBase
        The same axes instance, now with the wrap methods.
    """
    ax.__class__ = _axes_wrap_class(type(ax))
    assert isinstance(ax, AxesWrapBase)
    ax.set_wrap(
        wrapx,
        wrapy,
        set_lims=set_lims,
        seam_lines=seam_lines,
        margin=margin,
        seam_kwargs=seam_kwargs,
    )
    return ax
