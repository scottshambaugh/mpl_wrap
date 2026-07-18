# Changelog

## Future Work
### Features & Maintenance:

----

## [Unreleased]
### Added
### Changed
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
