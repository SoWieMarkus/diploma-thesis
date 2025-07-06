import argparse
import os
import logging
from typing import Callable
import cv2
import numpy

from gaze_estimator.visualization import draw_landmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

from tqdm import tqdm
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

get_face_detector_directory: Callable[[str, str], str] = (
    lambda video_output_dir, face_detector: os.path.join(
        video_output_dir, face_detector
    )
)

get_gaze_detector_directory: Callable[[str, str, str], str] = (
    lambda video_output_dir, face_detector, gaze_detector: os.path.join(
        video_output_dir, face_detector, gaze_detector
    )
)


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


def create_output_directories(video_output_directory: str) -> None:
    """
    Create output directories for each face detector and gaze detector results of the video.
    Args:
        video_output_directory(str): Path to the output directory of the video
    """
    if not os.path.exists(video_output_directory):
        os.makedirs(video_output_directory)

    for face_detector in FACE_DETECTORS:
        face_detector_directory = get_face_detector_directory(
            video_output_directory, face_detector.name
        )
        if not os.path.exists(face_detector_directory):
            os.makedirs(face_detector_directory)
        for gaze_detector in GAZE_DETECTORS:
            gaze_detector_directory = get_gaze_detector_directory(
                video_output_directory, face_detector.name, gaze_detector.name
            )
            if not os.path.exists(gaze_detector_directory):
                os.makedirs(gaze_detector_directory)


def process_frame(
    frame_number: int,
    frame: numpy.ndarray,
    meta: VideoCaptureMetaData,
    output_directory: str,
    args: argparse.Namespace,
):
    """
    Process a single frame to detect faces and estimate gaze.
    Args:
        frame (numpy.ndarray): The image frame to process.
        meta (VideoCaptureMetaData): Metadata about the video capture, such as resolution and frame rate.
        output_directory (str): Directory where the processed output will be saved.
    """

    # Store original frame as jpg
    cv2.imwrite(
        os.path.join(output_directory, f"{frame_number}.jpg"),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, args.image_quality],
    )

    for face_detector in FACE_DETECTORS:
        face = face_detector.detect_face(frame, meta)
        if face is None:
            continue

        # Draw detected landmarks to the frame and store it as jpg
        frame_with_landmarks = draw_landmarks(
            frame.copy(), face, face_detector.template_model, True
        )
        face_detector_output = get_face_detector_directory(
            output_directory, face_detector.name
        )
        cv2.imwrite(
            os.path.join(face_detector_output, f"{frame_number}.jpg"),
            frame_with_landmarks,
            [cv2.IMWRITE_JPEG_QUALITY, args.image_quality],
        )

        # TODO save frame with detected face

        for gaze_detector in GAZE_DETECTORS:
            # Normalize the input image for the gaze detector
            # Store the
            normalized_image = gaze_detector.normalize_input_image(
                frame.copy(), face, meta
            )
            output_file = get_gaze_detector_directory(
                output_directory,
                face_detector.name,
                gaze_detector.name,
            )
            cv2.imwrite(
                os.path.join(output_file, f"{frame_number}.jpg"),
                normalized_image,
                [cv2.IMWRITE_JPEG_QUALITY, args.image_quality],
            )
            gaze = gaze_detector.detect_gaze(face, normalized_image)

            # TODO save gaze to database


def process_video(video_path: str, args: argparse.Namespace):
    """
    Process a single video file and estimate gaze of every frame.
    Args:
        video_path (str): Path to the video file to process.
        args (argparse.Namespace): Command line arguments.
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return

    meta = get_meta_data_of_video(video)

    video_name = os.path.basename(video_path)

    logging.info(f"Processing video: {video_path}")

    output_directory = os.path.join(args.output, video_name)
    create_output_directories(output_directory)

    db_video = database.insert_video(video_name, video_path)
    # TODO clear entries of videos

    frame_count = 0
    with tqdm(total=meta.total_frames, desc=f"Processing {video_name}") as progress:
        while True:
            success, frame = video.read()
            if not success:
                logging.info(f"Finished processing video: {video_path}")
                break
            frame_count += 1
            progress.update(1)
            if frame_count % args.frame_step != 0:
                continue
            process_frame(frame_count, frame, meta, output_directory, args)


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
    parser.add_argument(
        "--image_quality",
        type=int,
        help="Quality of the saved images (0-100, where 100 is best quality)",
        default=30,
    )

    args = parser.parse_args()
    video_paths = load_videos(args.dir)
    logging.info(f"Found {len(video_paths)} videos in {args.dir}")

    for video_path in video_paths:
        process_video(video_path, args)


if __name__ == "__main__":
    main()
