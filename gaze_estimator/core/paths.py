"""Default locations of assets and data. All of them can be overridden by environment variables."""

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ASSETS_DIR = Path(os.environ.get("GAZE_ASSETS_DIR", REPOSITORY_ROOT / "data" / "assets"))
DATA_DIR = Path(os.environ.get("GAZE_DATA_DIR", REPOSITORY_ROOT / "data"))
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
PLOTS_DIR = DATA_DIR / "plots"
DEFAULT_DATABASE = Path(os.environ.get("GAZE_DATABASE", DATA_DIR / "gaze.db"))
WEB_DIST_DIR = REPOSITORY_ROOT / "web" / "dist"


def asset(name: str) -> Path:
    """Return the path of an asset file and fail with a helpful message if it is missing."""
    path = ASSETS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Asset '{name}' not found in {ASSETS_DIR}. Run `uv run gaze-download-assets` first."
        )
    return path
