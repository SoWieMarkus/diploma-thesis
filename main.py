import argparse
import os
import logging
import cv2
import numpy

from database import create_database_manager
from gaze_estimator.common.camera import get_meta_data_of_video, VideoCaptureMetaData
from gaze_estimator.face_detection.dlib_face_detector import DlibFaceDetector
from gaze_estimator.face_detection.face import Face
from gaze_estimator.face_detection.face_detector import FaceDetector
from gaze_estimator.face_detection.mediapipe_face_detector import MediapipeFaceDetector
from gaze_estimator.gaze_detection.eth_xgaze_detector import ETHXGazeDetector
from gaze_estimator.gaze_detection.gaze360_gaze_detector import Gaze360GazeDetector
from gaze_estimator.gaze_detection.gaze_detector import GazeDetector
from gaze_estimator.gaze_detection.head_rotation_gaze_detector import (
    HeadRotationGazeDetector,
)
from gaze_estimator.gaze_detection.mpiiface_gaze_detector import MPIIFaceGazeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Default directory where the videos to be processed are stored
INPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "input")

# Default directory where the processed output will be saved
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "output")

# supported video file extensions, for now its only mp4, but other formats should work as well
VIDEO_EXTENSIONS = [".mp4"]

FACE_DETECTORS: list[FaceDetector] = [
    MediapipeFaceDetector(),
    DlibFaceDetector(),
]

GAZE_DETECTORS: list[GazeDetector] = [
    ETHXGazeDetector(),
    HeadRotationGazeDetector(),
    Gaze360GazeDetector(),
    MPIIFaceGazeDetector(),
]

database = create_database_manager()


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


def process_frame(
    frame_number: int,
    frame: numpy.ndarray,
    meta: VideoCaptureMetaData,
    output_directory: str,
):
    """
    Process a single frame to detect faces and estimate gaze.
    Args:
        frame (numpy.ndarray): The image frame to process.
        meta (VideoCaptureMetaData): Metadata about the video capture, such as resolution and frame rate.
        output_directory (str): Directory where the processed output will be saved.
    """
    for face_detector in FACE_DETECTORS:
        face = face_detector.detect_face(frame, meta)
        # connect face to the frame
        if face is None:
            continue

        # TODO save frame with detected face

        for gaze_detector in GAZE_DETECTORS:

            normalized_image = gaze_detector.normalize_input_image(
                frame.copy(), face, meta
            )
            directory_name = f"{face_detector.name}_{gaze_detector.name}"
            output_file = os.path.join(
                output_directory, directory_name, f"{frame_number}.jpg"
            )
            cv2.imwrite(output_file, normalized_image)
            gaze = gaze_detector.detect_gaze(face, normalized_image)

            # TODO save gaze to database


def process_video(video_path: str, args: argparse.Namespace):
    """
    Process a single video file and estimate gaze of every frame.
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

    output_directory = os.path.join(
        args.output,
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    db_video = database.insert_video(meta.name, video_path)
    # TODO clear entries of videos

    frame_count = 0
    while True:
        success, frame = video.video_capture.read()
        if not success:
            logging.info(f"Finished processing video: {video_path}")
            break
        frame_count += 1
        if frame_count % args.frame_step != 0:
            continue
        process_frame(frame_count, frame, meta, output_directory)


def main():
    parser = argparse.ArgumentParser(description="Gaze Estimation")
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to directory containing the videos",
        default=INPUT_DIRECTORY,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to directory where the output will be saved",
        default=OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        help="Process every Nth frame (skip frames where frame_count %% N != 0)",
        default=3,
    )

    args = parser.parse_args()
    video_paths = load_videos(args.dir)
    logging.info(f"Found {len(video_paths)} videos in {args.dir}")

    for video_path in video_paths:
        process_video(video_path, args)


if __name__ == "__main__":
    main()
