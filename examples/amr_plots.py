import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from athenakit import AthenaData


def main(athdf_file: Path, level: int = 0) -> None:
    """Load AMR data and create example plots."""
    ad = AthenaData()
    ad.load(str(athdf_file))
    ad.config()

    # Slice plot of density
    fig, ax, *_ = ad.plot_slice(
        var="dens", level=level, axis="z", returnall=True, norm="log"
    )

    # Overlay meshblock boundaries
    for xmin, xmax, ymin, ymax, _, _ in ad.mb_geometry:
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec="white", lw=0.5
        )
        ax.add_patch(rect)
    ax.set_title(f"Density slice (level {level})")
    fig.savefig("density_slice.png", bbox_inches="tight")

    # Projection of eint weighted by density
    dens = ad.data("dens", dtype="uniform", level=level)
    eint = ad.data("eint", dtype="uniform", level=level)
    dz = ad.data("dz", dtype="uniform", level=level)
    proj = np.sum(eint * dens * dz, axis=0)

    edges = ad.get_slice_faces(level=level, axis="z")
    fig2, ax2 = plt.subplots()
    pcm = ax2.pcolormesh(edges["x"], edges["y"], proj, cmap="inferno", shading="auto")
    fig2.colorbar(pcm, ax=ax2, label="integrated eint*density")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    fig2.savefig("eint_projection.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python amr_plots.py <output.athdf> [level]")
    else:
        level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        main(Path(sys.argv[1]), level)
