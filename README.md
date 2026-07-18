# mpl_wrap

[![Builds](https://github.com/scottshambaugh/mpl_wrap/actions/workflows/builds.yml/badge.svg?branch=main)](https://github.com/scottshambaugh/mpl_wrap/actions/workflows/builds.yml)
[![Tests](https://github.com/scottshambaugh/mpl_wrap/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/scottshambaugh/mpl_wrap/actions/workflows/tests.yml)

Matplotlib helper library for plotting wrapped, angular, or periodic data.

**🚧 Under construction — not yet released.**

`mpl_wrap` plots continuous (unwrapped) data on axes that wrap around a periodic
window — angles on `(0, 360)`, time of day, phases — correctly routing lines,
bands, steps, and error bars across the seam instead of drawing jump artifacts.

## Installation

```
git clone https://github.com/scottshambaugh/mpl_wrap.git
cd mpl_wrap
uv sync --group dev
```
