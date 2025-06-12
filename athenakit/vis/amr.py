"""AMR visualization helpers."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from ..athena_data import AthenaData, asnumpy


_DataSource = Union[AthenaData, Dict[str, Any]]


def _as_data_dict(src: _DataSource) -> Dict[str, Any]:
    """Return a dictionary-like view of AMR data.

    Parameters
    ----------
    src : AthenaData or dict
        Input data source. If ``AthenaData`` is provided, the returned
        dictionary contains numpy arrays extracted from the object.
    """
    if isinstance(src, dict):
        return src

    if isinstance(src, AthenaData):
        dic = {
            "var_names": list(src.data_raw.keys()),
            "mb_list": asnumpy(src.mb_list),
            "mb_data": {v: asnumpy(src.data_raw[v]) for v in src.data_raw},
            "mb_geometry": asnumpy(src.mb_geometry),
            "x1min": float(src.x1min),
            "x1max": float(src.x1max),
            "x2min": float(src.x2min),
            "x2max": float(src.x2max),
            "x3min": float(src.x3min),
            "x3max": float(src.x3max),
        }
        return dic

    raise TypeError("Unsupported data source")


def plot_amr_slice_patchwork(
    data: _DataSource,
    variable: str = "dens",
    slice_axis: str = "z",
    slice_position: float = 0.0,
    log_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: Tuple[int, int] = (10, 8),
    show_grid: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a slice plot by drawing each mesh block as a separate patch.

    Parameters
    ----------
    data : AthenaData or dict
        Data source containing ``mb_list``, ``mb_data`` and ``mb_geometry``.
    variable : str
        Variable name to plot (``'dens'``, ``'eint'``, etc.).
    slice_axis : str
        Axis to slice through (``'x'``, ``'y'`` or ``'z'``).
    slice_position : float
        Position along slice axis (in code units).
    log_scale : bool
        Whether to use logarithmic scaling.
    vmin, vmax : float, optional
        Color scale limits.
    figsize : tuple
        Figure size.
    show_grid : bool
        Whether to show grid lines on the plot.
    """

    if slice_axis not in {"x", "y", "z"}:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'")

    data_dic = _as_data_dict(data)

    if variable not in data_dic["var_names"]:
        raise ValueError(
            f"Variable '{variable}' not found. Available: {data_dic['var_names']}"
        )

    axis_map = {"x": 0, "y": 1, "z": 2}
    slice_idx = axis_map[slice_axis]
    plot_axes = [i for i in range(3) if i != slice_idx]
    axis_names = ["x", "y", "z"]

    fig, ax = plt.subplots(figsize=figsize)

    mesh_bounds = np.array(
        [
            data_dic["x1min"],
            data_dic["x1max"],
            data_dic["x2min"],
            data_dic["x2max"],
            data_dic["x3min"],
            data_dic["x3max"],
        ]
    ).reshape(3, 2)

    valid_blocks = []
    data_min = np.inf
    data_max = -np.inf

    for mb_id in data_dic["mb_list"]:
        bounds = np.array(data_dic["mb_geometry"][mb_id]).reshape(3, 2)
        if not (bounds[slice_idx, 0] <= slice_position < bounds[slice_idx, 1]):
            continue

        block_data = np.asarray(data_dic["mb_data"][variable][mb_id])
        slice_coord = np.linspace(
            bounds[slice_idx, 0], bounds[slice_idx, 1], block_data.shape[slice_idx]
        )
        slice_block_idx = np.argmin(np.abs(slice_coord - slice_position))

        if slice_axis == "z":
            slice_data = block_data[slice_block_idx, :, :]
        elif slice_axis == "y":
            slice_data = block_data[:, slice_block_idx, :]
        else:
            slice_data = block_data[:, :, slice_block_idx]

        plot_data = np.log10(slice_data) if log_scale else slice_data

        dx = (
            show_grid
            * abs(mesh_bounds[plot_axes[0], 1] - mesh_bounds[plot_axes[0], 0])
            / 1000
        )
        dy = (
            show_grid
            * abs(mesh_bounds[plot_axes[1], 1] - mesh_bounds[plot_axes[1], 0])
            / 1000
        )
        extent = [
            bounds[plot_axes[0], 0] + dx,
            bounds[plot_axes[0], 1] - dx,
            bounds[plot_axes[1], 0] + dy,
            bounds[plot_axes[1], 1] - dy,
        ]

        data_min = min(data_min, np.min(plot_data))
        data_max = max(data_max, np.max(plot_data))
        valid_blocks.append({"plot_data": plot_data, "extent": extent})

    if not valid_blocks:
        raise ValueError("Slice does not intersect with any mesh block")

    vmin = data_min if vmin is None else vmin
    vmax = data_max if vmax is None else vmax

    for blk in valid_blocks:
        im = ax.imshow(
            blk["plot_data"],
            extent=blk["extent"],
            origin="lower",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_xlim(mesh_bounds[plot_axes[0], 0], mesh_bounds[plot_axes[0], 1])
    ax.set_ylim(mesh_bounds[plot_axes[1], 0], mesh_bounds[plot_axes[1], 1])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"log10({variable})" if log_scale else variable)

    ax.set_xlabel(f"{axis_names[plot_axes[0]]} [code units]")
    ax.set_ylabel(f"{axis_names[plot_axes[1]]} [code units]")
    ax.set_title(f"{variable} slice at {slice_axis}={slice_position}")
    plt.tight_layout()
    return fig, ax


def plot_amr_projection(
    ad: AthenaData,
    variable: str = "eint",
    weight: str = "dens",
    axis: str = "z",
    level: int = 0,
    log_scale: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Make a projection of ``variable`` weighted by ``weight`` along an axis.

    Parameters
    ----------
    ad : AthenaData
        Loaded data object.
    variable : str
        Variable to project.
    weight : str
        Weight variable.
    axis : str
        Projection axis (``'x'``, ``'y'`` or ``'z'``).
    level : int
        AMR level to use for uniform data.
    log_scale : bool
        Whether to show logarithmic color scale.
    """

    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map[axis]
    other_axes = [a for a in ["x", "y", "z"] if a != axis]

    data = ad.data(variable, dtype="uniform", level=level)
    wgt = ad.data(weight, dtype="uniform", level=level)
    dlen = ad.data(f"d{axis}", dtype="uniform", level=level)

    proj = np.sum(data * wgt * dlen, axis=idx)

    edges = ad.get_slice_faces(level=level, axis=axis)
    x_edges = edges[other_axes[0]]
    y_edges = edges[other_axes[1]]

    plot_data = np.log10(proj) if log_scale else proj

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        plot_data,
        cmap="inferno",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(pcm, ax=ax, label=f"integrated {variable}*{weight}")
    ax.set_xlabel(other_axes[0])
    ax.set_ylabel(other_axes[1])
    ax.set_aspect("equal")
    ax.set_title(f"Projection of {variable} weighted by {weight}")
    plt.tight_layout()
    return fig, ax
