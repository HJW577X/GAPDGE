"""Command-line entry point for GAPDGE."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from gapdge.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
