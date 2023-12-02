import numpy

from .face_template_model import FaceTemplateModel


class MediapipeFaceTemplateModel(FaceTemplateModel):
    def __init__(self):
        super().__init__("template_model_3d_mediapipe.csv")

        if len(self.landmarks) != 468:
            raise ValueError(
                "Something went wrong while reading the 3D template model. Its supposed to contain 468 face landmarks."
            )

    def get_right_eye_ear_indices(self):
        return numpy.array([33, 160, 158, 133, 153, 144], dtype="int")

    def get_left_eye_ear_indices(self):
        return numpy.array([362, 385, 387, 263, 373, 380], dtype="int")

    def get_right_eye_indices(self):
        return numpy.array([33, 133], dtype="int")

    def get_left_eye_indices(self):
        return numpy.array([362, 263], dtype="int")

    def get_mouth_indices(self):
        return numpy.array([78, 308], dtype="int")

    def get_nose_indices(self):
        return numpy.array([240, 460], dtype="int")

    def get_nose_tip_index(self):
        return 1

    def get_chin_index(self):
        return 199
