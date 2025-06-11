from gaze_estimator import estimate_gaze
import argparse
import os
import logging
import cv2

from gaze_estimator.common.camera import get_meta_data_of_video
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


def process_videos(video_paths: list[str], args: argparse.Namespace):
    """
    Process each video file in the list and estimate gaze.
    Args:
        video_paths (list[str]): List of video file paths to process.
    """
    for video_path in video_paths:
        logging.info(f"Processing video: {video_path}")

        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logging.error(f"Could not open video file: {video}")
            continue

        meta = get_meta_data_of_video(video)

        # TODO delete the face detection and gaze detection from the video inside the database
        # TODO create video if it does not exist in the database
        while True:
            success, frame = video.video_capture.read()

            if not success:
                break

            # TODO create frame in the database if it doesnt exist
            # maybe cache and store afterwards with bulk insert

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

                    # TODO connect gaze to fhe face in the database


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
    videos = load_videos(args.dir)
    logging.info(f"Found {len(videos)} videos in {args.dir}")

    process_videos(videos, args)


if __name__ == "__main__":
    main()
