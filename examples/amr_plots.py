import sys
from pathlib import Path

import matplotlib.pyplot as plt

from athenakit import AthenaData
from athenakit.vis import plot_amr_slice_patchwork, plot_amr_projection


def main(athdf_file: Path, level: int = 0) -> None:
    """Load AMR data and create example plots."""
    ad = AthenaData()
    ad.load(str(athdf_file))
    ad.config()

    # Slice plot of density with meshblock boundaries
    fig, ax = plot_amr_slice_patchwork(ad, variable="dens", slice_axis="z")
    for xmin, xmax, ymin, ymax, _, _ in ad.mb_geometry:
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec="white", lw=0.5
        )
        ax.add_patch(rect)
    fig.savefig("density_slice.png", bbox_inches="tight")

    # Projection of eint weighted by density
    fig2, _ = plot_amr_projection(ad, variable="eint", weight="dens", axis="z", level=level)
    fig2.savefig("eint_projection.png", bbox_inches="tight")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python amr_plots.py <output.athdf> [level]")
    else:
        level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        main(Path(sys.argv[1]), level)
