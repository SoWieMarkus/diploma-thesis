import numpy

from .gaze_model import GazeModel
from ...utils.math import convert_rotation_vector_to_matrix


# NO AI JUST GEOMETRY!!!

class HeadRotationModel(GazeModel):

    def __init__(self):
        super().__init__(
            name="head-rotation",
            download_url=None,
            file_name=None
        )

    def load_model(self):
        pass

    def detect_gaze(self, image, face, video, face_model_name, frame_number):
        # base vector orthogonal to camera plane defined as z = 1
        # vector is pointing to the camera plane
        original_vector = numpy.array([0, 0, -1])

        # convert rotation vector to rotation matrix
        matrix = convert_rotation_vector_to_matrix(face.face_head_rotation_vector)

        # rotate vector to align with the rotation of the face
        transformed_vector = matrix @ original_vector
        return 0, 0, transformed_vector / numpy.linalg.norm(transformed_vector)
