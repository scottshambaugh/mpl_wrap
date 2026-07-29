# Changelog

## Future Work
### Features & Maintenance:

----

## [Unreleased]
### Added
### Changed
* Fill edges are stroked from the wrapped band boundary itself, removing the
  stray seam line drawn where a band stretch spanning a full period borders
  narrower stretches on both sides
### Removed

## [0.2.0] - 2026-07-27
### Added
* `step_wrapped`, `fill_betweenx_wrapped`, `hlines_wrapped`, `vlines_wrapped`,
  `axhspan_wrapped`, and `axvspan_wrapped` helpers, mirroring their `ax.*`
  counterparts, also available as `AxesWrap` methods. Spans and line spans are
  split at the seam, and a span of a period or more sweeps the window.
* The mirrored methods' own arguments: `where`, `interpolate`, `step` and a
  default `y2=0` on `fill_between_wrapped`, and `orientation`, `baseline` and
  `fill` on `stairs_wrapped` (`baseline` and `fill` were silently dropped before)
* `py.typed` marker (PEP 561), so the annotations are visible to type checkers
  in downstream projects

### Changed
* Every helper returns the artist type its matplotlib counterpart does, in the
  same container: `fill_between_wrapped` a `FillBetweenPolyCollection` and
  `stairs_wrapped` a `StepPatch`, both carrying the wrapped geometry and
  re-wrapping on `set_data`. `stairs_wrapped` returns one artist for the line
  and filled forms, so its `**kwargs` are now patch rather than line properties.
* Markers are drawn only at the data points, not at the vertices that seam
  routing and step expansion insert
* Wrapped fills follow `ax.fill_between` more closely: the fill colour cycle,
  an edge stroked along the band rather than along the tiling seams, no spill
  outside the axes, and non-finite samples breaking the band instead of raising
* `errorbar_wrapped` matches `ax.errorbar`: cap size, the `errorbar.capsize` and
  `errorbar.elinewidth` rcParams, x errors before y errors in the container, the
  data line's label and zorder, and a data line that follows the wrapped
  polyline instead of jumping at the seam
* `stairs_wrapped` defaults to `baseline=0` as `ax.stairs` does, so both ends
  drop to the baseline. Pass `baseline=None` for the previous bare staircase.

### Removed

## [0.1.0] - 2026-07-18
Initial release!

### Added
* `plot_wrapped`, `scatter_wrapped`, `fill_between_wrapped`, `stairs_wrapped`,
  and `errorbar_wrapped` helpers for plotting continuous (unwrapped) data on
  wrapped/periodic axes, with `wrapx`/`wrapy` (min, max) windows and datetime support
* `set_wrap` helper to store wrap windows on an axes (picked up by the helpers
  by default), optionally setting axis limits and drawing seam lines
* `AxesWrap` class with the helpers as methods, available as the `"wrap"`
  projection, and `wrap_axes` to upgrade an existing axes of any projection in
  place
* `wrap_line` and `wrap_points` data-processing functions that return the
  wrapped arrays without plotting, also available as `AxesWrap` methods
