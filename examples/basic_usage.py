import sys
from pathlib import Path
from athenakit import io, kit


def main(run_dir: Path) -> None:
    """Demonstrate basic conversion and history loading."""
    bin_dir = run_dir / "bin"
    athdf_dir = run_dir / "athdf"
    athdf_dir.mkdir(exist_ok=True)

    # Convert all binary dumps
    kit.bins_to_athdfs(str(bin_dir), str(athdf_dir), info=True)

    # Read and print history time column
    hist_file = run_dir / "hist" / "history.hst"
    if hist_file.exists():
        hist = io.athena_read.hst(str(hist_file))
        print("Time steps:", hist["time"])
    else:
        print(f"History file {hist_file} not found")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python basic_usage.py <run_directory>")
    else:
        main(Path(sys.argv[1]))
