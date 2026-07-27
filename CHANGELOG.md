# Changelog

## Future Work
### Features & Maintenance:

----

## [Unreleased]
### Added
* `py.typed` marker (PEP 561), so the annotations are visible to type checkers
  in downstream projects
* `step_wrapped`, mirroring `ax.step` (n x-values against n y-values plus a
  `where` policy), also available as an `AxesWrap` method
* `fill_between_wrapped` now supports `ax.fill_between`'s `where`,
  `interpolate`, and `step` arguments, and defaults `y2` to 0
* `stairs_wrapped` now supports `ax.stairs`' `orientation`, `baseline`, and
  `fill` arguments (previously `baseline` and `fill` were silently dropped)
### Changed
* `stairs_wrapped` defaults to `baseline=0` as `ax.stairs` does, so both ends of
  the staircase now drop to the baseline. Pass `baseline=None` for the previous
  bare-staircase behavior.
* `fill_between_wrapped` breaks the band at non-finite samples instead of
  raising, so gappy series fill run-by-run
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
