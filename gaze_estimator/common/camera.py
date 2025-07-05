import cv2
from dataclasses import dataclass
from fractions import Fraction
import numpy


@dataclass
class VideoCaptureMetaData:
    width: int
    height: int
    frame_rate: int
    total_frames: int
    camera_matrix: numpy.array
    distortion_coefficients: numpy.array
    aspect_ratio: Fraction


def get_meta_data_of_video(video_capture: cv2.VideoCapture) -> VideoCaptureMetaData:
    """
    Get metadata of a video capture object.
    Args:
        video_capture (cv2.VideoCapture): The video capture object.
    Returns:
        VideoCaptureMetaData: A dataclass containing the metadata of the video.
    """
    if not video_capture.isOpened():
        raise RuntimeError("Video capture is not opened.")

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    center = (width / 2, height / 2)
    camera_matrix = numpy.array(
        [
            [width, 0.0, center[0]],
            [0.0, width, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype="double",
    )
    aspect_ratio = Fraction(width, height).limit_denominator()

    return VideoCaptureMetaData(
        width=width,
        height=height,
        frame_rate=frame_rate,
        total_frames=total_frames,
        camera_matrix=camera_matrix,
        distortion_coefficients=numpy.zeros((4, 1), dtype="double"),
        aspect_ratio=aspect_ratio,
    )
