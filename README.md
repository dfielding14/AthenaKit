# AthenaKit

Toolkit for research with AthenaK

## Overview

The code is designed to analyze and visualize the simulation data dumped by [AthenaK](https://github.com/IAS-Astrophysics/athenak) conveniently and efficiently.

Current features are:
- Enables *performance-portability*. Support both CPU and CUDA GPU
- Support `MPI`
- Non-relativistic (Newtonian) hydrodynamics and MHD
- General relativistic (GR) hydrodynamics and MHD in stationary spacetimes

## Getting started

Documentation is under construction on the [wiki](https://github.com/mh-guo/AthenaKit/wiki) pages.

## Basic Usage

The `examples` directory contains short scripts demonstrating common tasks.
To convert binary dumps to ATHDF format and read a history file:

```bash
python examples/basic_usage.py /path/to/run
```

This will create `.athdf` files next to the binaries and print the time
column from the corresponding history file.

## AMR Visualization

To visualize adaptive mesh refinement (AMR) data, use the
`examples/amr_plots.py` script. It loads an `.athdf` file, produces a slice
plot of density with meshblock boundaries and then makes a projection of
internal energy weighted by density.

```bash
python examples/amr_plots.py output.athdf [level]
```

The optional `level` argument selects the uniform AMR level for the projection
plot (default `0`). The script saves `density_slice.png` and
`eint_projection.png` in the current directory.
