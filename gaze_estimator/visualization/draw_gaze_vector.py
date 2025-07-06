import numpy
import cv2
from ..face_detection import Face
from ..gaze_detection import Gaze
from ..common.camera import VideoCaptureMetaData


def project_points(points3d: numpy.ndarray, meta: VideoCaptureMetaData):
    """
    Projects 3D points to 2D image coordinates using camera parameters.
    Args:
        points3d (numpy.ndarray): Array of 3D points to be projected.
        meta (VideoCaptureMetaData): Metadata containing camera parameters.
    Returns:
        numpy.ndarray: Array of 2D points projected onto the image plane.
    """
    rvec = numpy.zeros(3, dtype=float)
    tvec = numpy.zeros(3, dtype=float)

    points2d, _ = cv2.projectPoints(
        points3d, rvec, tvec, meta.camera_matrix, meta.distortion_coefficients
    )
    return points2d.reshape(-1, 2)


def draw_gaze_vector(
    frame: numpy.ndarray, face: Face, gaze: Gaze, meta: VideoCaptureMetaData
) -> numpy.ndarray:
    """
    Draws the gaze vector on the given frame.
    Args:
        frame (numpy.ndarray): The image frame on which to draw the gaze vector.
        face (Face): The detected face object containing the center point.
        gaze (Gaze): The gaze vector to be drawn.
        meta (VideoCaptureMetaData): Metadata for camera parameters.
    Returns:
        numpy.ndarray: The frame with the gaze vector drawn.
    """
    point1 = face.center
    point2 = face.center + 0.2 * gaze.vector

    points3d = numpy.vstack([point1, point2])
    points2d = project_points(points3d, meta)

    pt0 = tuple(numpy.round(points2d[0]).astype(int).tolist())
    pt1 = tuple(numpy.round(points2d[1]).astype(int).tolist())

    color = (255, 0, 255)
    cv2.line(frame, pt0, pt1, color, 2, cv2.LINE_AA)
    return frame
