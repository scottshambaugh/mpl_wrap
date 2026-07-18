"""Helper functions for plotting wrapped, angular, or periodic data on matplotlib axes.

The public API will provide free functions that mirror the core matplotlib
plotting methods, taking an ``Axes`` plus ``wrapx`` / ``wrapy`` (min, max)
windows: ``plot_wrapped``, ``scatter_wrapped``, ``fill_between_wrapped``,
``stairs_wrapped``, ``errorbar_wrapped``, and a ``set_wrap`` setup helper.
"""
