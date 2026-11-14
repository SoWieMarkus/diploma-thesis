"""Live webcam demo: draws landmarks, head pose and gaze of one face detector + gaze model."""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np

from .. import visualization as vis
from ..core.camera import Camera
from ..core.geometry import Display
from ..faces.registry import FACE_DETECTORS, create_face_detector
from ..gaze.registry import GAZE_ESTIMATORS, create_gaze_estimator
from .common import setup_logging

logger = logging.getLogger("gaze-demo")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", choices=list(FACE_DETECTORS), default="mediapipe")
    parser.add_argument("--gaze", choices=list(GAZE_ESTIMATORS), default="eth-xgaze")
    parser.add_argument("--camera", default="0", help="Camera index or path/URL of a video.")
    parser.add_argument("--profile", type=Path, help="Camera profile YAML (intrinsics).")
    parser.add_argument("--width", type=int, help="Requested capture width.")
    parser.add_argument("--height", type=int, help="Requested capture height.")
    parser.add_argument(
        "--display",
        type=float,
        nargs=3,
        metavar=("WIDTH", "HEIGHT", "OFFSET"),
        default=(0.07, 0.15, 0.01),
        help="Virtual display in meters for the 'looking at display' indicator.",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror the preview like a selfie.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.verbose)

    source = int(args.camera) if args.camera.isdigit() else args.camera
    capture = cv2.VideoCapture(source)
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera '{args.camera}'.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera = Camera.resolve(width, height, args.profile)
    logger.info("Camera %dx%d, horizontal FOV %.1f deg", width, height, camera.hfov_deg)

    face_detector = create_face_detector(args.face)
    gaze_estimator = create_gaze_estimator(args.gaze)
    display = Display(*args.display)
    show_landmarks = True
    window = f"Gaze demo | {args.face} + {args.gaze}  (q: quit, l: landmarks)"

    last = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                logger.error("Could not read a frame from the camera.")
                break

            face = face_detector.detect(frame, camera)
            lines = [f"{args.face} + {args.gaze}"]
            looking = None
            if face is not None:
                gaze = gaze_estimator.estimate(frame, face, camera)
                looking = bool(display.hits(face.center, gaze.vector))
                angle = float(display.angular_distance(face.center, gaze.vector)[0])
                if show_landmarks:
                    vis.draw_face(frame, face, face_detector.template)
                vis.draw_head_axes(frame, face, camera)
                vis.draw_gaze(frame, face, gaze, camera)
                lines += [
                    f"pitch {math.degrees(gaze.pitch):6.1f}  yaw {math.degrees(gaze.yaw):6.1f} deg",
                    f"distance {face.distance:.2f} m   EAR {face.ear:.2f}",
                    f"angle to display {math.degrees(angle):5.1f} deg",
                ]
            else:
                lines.append("no face")

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            lines.append(f"{fps:4.1f} fps")

            if args.mirror:
                frame = np.ascontiguousarray(frame[:, ::-1])
            vis.draw_text_block(frame, lines)
            vis.draw_status(frame, looking)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("l"):
                show_landmarks = not show_landmarks
    finally:
        capture.release()
        face_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
