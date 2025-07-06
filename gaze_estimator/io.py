import os
from typing import Callable
from .face_detection.face_detector import FaceDetector
from .gaze_detection.gaze_detector import GazeDetector


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

# supported video file extensions, for now its only mp4, but other formats should work as well
VIDEO_EXTENSIONS = [".mp4"]


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


def create_output_directories(
    video_output_directory: str,
    face_detectors: list[FaceDetector],
    gaze_detectors: list[GazeDetector],
) -> None:
    """
    Create output directories for each face detector and gaze detector results of the video.
    Args:
        video_output_directory(str): Path to the output directory of the video
    """
    if not os.path.exists(video_output_directory):
        os.makedirs(video_output_directory)

    for face_detector in face_detectors:
        face_detector_directory = get_face_detector_directory(
            video_output_directory, face_detector.name
        )
        if not os.path.exists(face_detector_directory):
            os.makedirs(face_detector_directory)
        for gaze_detector in gaze_detectors:
            gaze_detector_directory = get_gaze_detector_directory(
                video_output_directory, face_detector.name, gaze_detector.name
            )
            if not os.path.exists(gaze_detector_directory):
                os.makedirs(gaze_detector_directory)
