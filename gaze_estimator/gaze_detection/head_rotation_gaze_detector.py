import cv2
from .gaze_detector import GazeDetector
from .gaze import Gaze
import numpy


class HeadRotationGazeDetector(GazeDetector):

    def __init__(self):
        super().__init__(name="head-rotation")

    def load_model(self):
        return super().load_model()

    def normalize_input_image(self, frame, face, meta):
        # Since we have no normalization for head rotation gaze detection,
        # we override this method and just return the original frame.
        return frame

    def detect_gaze(self, face, normalized_frame):
        # Base vector orthogonal to camera plane defined as z = 1
        # Vector is pointing to the camera plane
        original_vector = numpy.array([0, 0, -1])

        # convert rotation vector to rotation matrix
        matrix = cv2.Rodrigues(face.head_rotation_vector)[0]

        # Rotate vector to align with the rotation of the face
        transformed_vector = matrix @ original_vector
        return Gaze(
            vector=transformed_vector / numpy.linalg.norm(transformed_vector),
            pitch=0,
            yaw=0,
        )
