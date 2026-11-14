"""Process all videos of a folder: every n-th frame with all face detectors and gaze models.

Results are written to a SQLite database and the processed frames are stored as JPGs. Running the
command again only computes what is missing (new videos, new models or a smaller step).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..core.paths import DEFAULT_DATABASE, INPUT_DIR, OUTPUT_DIR
from ..db.session import create_db_engine
from ..faces import registry as faces
from ..gaze import registry as gazes
from ..pipeline.processor import ProcessOptions, Processor, find_videos
from .common import setup_logging

logger = logging.getLogger("gaze-process")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DIR, help="Folder with the videos.")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR, help="Folder for the frame JPGs."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE, help="SQLite database file.")
    parser.add_argument("--step", type=int, default=5, help="Process every n-th frame.")
    parser.add_argument(
        "--faces",
        nargs="+",
        default=["all"],
        help=f"Face detectors ({', '.join(faces.FACE_DETECTORS)}).",
    )
    parser.add_argument(
        "--gazes",
        nargs="+",
        default=["all"],
        help=f"Gaze models ({', '.join(gazes.GAZE_ESTIMATORS)}).",
    )
    parser.add_argument("--profile", type=Path, help="Camera profile YAML for all videos.")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the selected face detectors (and their gazes). Frame labels are kept.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)
    if args.step < 1:
        raise SystemExit("--step must be >= 1")

    face_names = faces.resolve_names(args.faces)
    gaze_names = gazes.resolve_names(args.gazes)
    videos = find_videos(args.input)
    if not videos:
        raise SystemExit(f"No videos found in '{args.input}'.")

    logger.info(
        "%d video(s) | faces: %s | gazes: %s | every %d. frame",
        len(videos),
        ", ".join(face_names),
        ", ".join(gaze_names),
        args.step,
    )

    engine = create_db_engine(args.db)
    processor = Processor(
        engine,
        args.db,
        args.output,
        face_names,
        gaze_names,
        ProcessOptions(
            step=args.step,
            jpeg_quality=args.jpeg_quality,
            force=args.force,
            camera_profile=args.profile,
        ),
    )
    try:
        for video in videos:
            stats = processor.process_video(video)
            logger.info(
                "%s: %d new frames, %d new faces, %d new gazes",
                video.name,
                stats["frames"],
                stats["faces"],
                stats["gazes"],
            )
    finally:
        processor.close()
    logger.info("Done. Database: %s", args.db.resolve())


if __name__ == "__main__":
    main()
