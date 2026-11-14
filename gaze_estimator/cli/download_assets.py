"""Download the model weights that are missing in ``data/assets``."""

from __future__ import annotations

import argparse
import bz2
import logging
import shutil
import urllib.request
from pathlib import Path

from ..core.paths import ASSETS_DIR
from .common import setup_logging

logger = logging.getLogger("gaze-download-assets")

ASSETS: dict[str, str] = {
    "face-model-dlib.dat": "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    "face-landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "gaze-model-ethx.pth": "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth",
    "gaze-model-mpiifacegaze.pth": "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiifacegaze_resnet_simple.pth",
    "gaze-model-gaze360.pkl": "https://drive.google.com/uc?id=18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ&export=download",
}


def download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "gaze-estimator"})
    with urllib.request.urlopen(request) as response, open(temporary, "wb") as file:
        if url.endswith(".bz2"):
            shutil.copyfileobj(bz2.BZ2File(response), file)
        else:
            shutil.copyfileobj(response, file)
    temporary.replace(target)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download even if the file exists.")
    args = parser.parse_args(argv)
    setup_logging()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, url in ASSETS.items():
        target = ASSETS_DIR / name
        if target.is_file() and not args.force:
            logger.info("%-28s already present", name)
            continue
        logger.info("%-28s downloading from %s", name, url)
        try:
            download(url, target)
        except Exception as error:  # noqa: BLE001 - report and continue with the next file
            logger.error("%-28s failed: %s", name, error)
            failed.append(name)
    if failed:
        raise SystemExit(
            "Some downloads failed (Google Drive sometimes blocks scripted downloads). "
            f"Download them manually into {ASSETS_DIR}: {', '.join(failed)}"
        )


if __name__ == "__main__":
    main()
