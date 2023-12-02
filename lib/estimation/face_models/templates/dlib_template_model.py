import numpy

from .face_template_model import FaceTemplateModel


class DlibFaceTemplateModel(FaceTemplateModel):

    def __init__(self):
        super().__init__("template_model_3d_dlib.csv")

        if len(self.landmarks) != 68:
            raise ValueError(
                "Something went wrong while reading the 3D template model. Its supposed to contain 68 face landmarks."
            )

    def get_right_eye_ear_indices(self):
        return numpy.array([36, 37, 38, 39, 40, 41], dtype="int")

    def get_left_eye_ear_indices(self):
        return numpy.array([42, 43, 44, 45, 46, 47], dtype="int")

    def get_right_eye_indices(self):
        return numpy.array([36, 39], dtype="int")

    def get_left_eye_indices(self):
        return numpy.array([42, 45], dtype="int")

    def get_mouth_indices(self):
        return numpy.array([48, 54], dtype="int")

    def get_nose_indices(self):
        return numpy.array([31, 35], dtype="int")

    def get_nose_tip_index(self):
        return 30

    def get_chin_index(self):
        return 8
