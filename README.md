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

## Project Structure

The library is implemented as a collection of packages under `athenakit`:

* **athena_data.py** – high level interface for loading data dumps.  It can read
  binary `.bin` outputs, converted `.athdf` files or previously saved HDF5
  objects and exposes helpers for configuration and plotting.
* **io** – routines for reading and writing AthenaK formats including history
  files, binary dumps and ATHDF outputs.
* **kit.py** – utility functions used by the examples for converting data and
  performing common mathematical operations.
* **physics** – domain specific helper functions such as GRMHD metric support.
* **vis** – minimal visualisation helpers built on Matplotlib.
* **app** – small analysis scripts for specialised workflows.

Older standalone modules are kept inside the `python` directory for reference.

## Running the examples

The `examples` directory shows how to perform common analysis tasks.

### basic_usage.py

```bash
python examples/basic_usage.py /path/to/run
```

This converts all binary dumps under `/path/to/run/bin` to `.athdf` files inside
`/path/to/run/athdf` and then prints the list of time steps read from the
corresponding history file.

## AMR Visualization

To visualize adaptive mesh refinement (AMR) data, use the
`examples/amr_plots.py` script. It loads an `.athdf` file, produces a slice
plot of density with meshblock boundaries and then makes a projection of
internal energy weighted by density.

The optional `level` argument selects the uniform AMR level for the projection
plot (default `0`). The script saves `density_slice.png` and
`eint_projection.png` in the current directory.

## CUDA and MPI Support

AthenaKit automatically detects optional dependencies for acceleration. If
the [CuPy](https://cupy.dev) package is available the internal arrays are
backed by CUDA memory, while [mpi4py](https://mpi4py.readthedocs.io)
enables multi-process execution.

### Enabling GPU Acceleration

Install a CuPy build that matches your CUDA toolkit, for example:

```bash
pip install cupy-cuda12x  # for CUDA 12
```

When imported, the package prints `CuPy enabled` if GPU support is active.
All array calculations then run on the GPU transparently.

### Running with MPI

After installing `mpi4py` you can launch the same scripts with `mpirun`
to distribute workloads across processes:

```bash
pip install mpi4py
mpirun -n 4 python examples/basic_usage.py /path/to/run
```

Parallel reading of ATHDF files is supported. Writing requires a
parallel-enabled HDF5 installation and may fall back to serial mode if
not available.

## Development and Testing

This repository uses `pytest` for its unit tests.  After installing the
dependencies run

```bash
pytest
```

to make sure the utility functions continue to work.

Further documentation is being developed on the
[wiki](https://github.com/mh-guo/AthenaKit/wiki) pages.
