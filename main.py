import argparse
import os
import logging
import cv2
import numpy

from gaze_estimator.common.camera import get_meta_data_of_video, VideoCaptureMetaData
from gaze_estimator.face_detection.dlib_face_detector import DlibFaceDetector
from gaze_estimator.face_detection.face import Face
from gaze_estimator.face_detection.mediapipe_face_detector import MediapipeFaceDetector
from gaze_estimator.gaze_detection.eth_xgaze_detector import ETHXGazeDetector
from gaze_estimator.gaze_detection.gaze360_gaze_detector import Gaze360GazeDetector
from gaze_estimator.gaze_detection.head_rotation_gaze_detector import (
    HeadRotationGazeDetector,
)
from gaze_estimator.gaze_detection.mpiiface_gaze_detector import MPIIFaceGazeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Default directory where the videos to be processed are stored
VIDEO_DIRECTORY = "./data/input"

# supported video file extensions, for now its only mp4, but other formats should work as well
VIDEO_EXTENSIONS = [".mp4"]

FACE_DETECTORS = [
    MediapipeFaceDetector(),
    DlibFaceDetector(),
]

GAZE_DETECTORS = [
    ETHXGazeDetector(),
    HeadRotationGazeDetector(),
    Gaze360GazeDetector(),
    MPIIFaceGazeDetector(),
]


def load_videos(directory: str) -> list[str]:
    """
    Get all video files from the specified directory.
    Args:
        directory (str): Path to the directory containing video files.
    Returns:
        list[str]: List of video file paths inside this directory.
    """
    return [
        f"{directory}/{file}"
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file))
        and os.path.splitext(file)[1].lower() in VIDEO_EXTENSIONS
    ]


def process_frame(frame: numpy.ndarray, meta: VideoCaptureMetaData) -> list[Face]:

    for face_detector in FACE_DETECTORS:
        face = face_detector.detect_face(frame, meta)
        # connect face to the frame
        if face is None:
            continue

        # TODO save frame with detected face
        for gaze_detector in GAZE_DETECTORS:
            # TODO normalize image
            # TODO save normalized image
            gaze = gaze_detector.detect_gaze(face, frame)


def process_video(video_path: str, args: argparse.Namespace):
    """
    Process a single video file and estimate gaze.
    Args:
        video_path (str): Path to the video file to process.
        args (argparse.Namespace): Command line arguments.
    """
    logging.info(f"Processing video: {video_path}")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return

    meta = get_meta_data_of_video(video)

    logging.info(f"Processing video: {video_path}")

    # TODO delete the face detection and gaze detection from the video inside the database
    # TODO create video if it does not exist in the database
    while True:
        success, frame = video.video_capture.read()
        if not success:
            break
        process_frame(frame, meta)


def main():
    parser = argparse.ArgumentParser(description="Gaze Estimation")
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to directory containing the videos",
        default=VIDEO_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to directory where the output will be saved",
    )

    args = parser.parse_args()
    video_paths = load_videos(args.dir)
    logging.info(f"Found {len(video_paths)} videos in {args.dir}")

    for video_path in video_paths:
        process_video(video_path, args)


if __name__ == "__main__":
    main()
