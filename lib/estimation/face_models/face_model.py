from abc import abstractmethod

import numpy

from ..model import Model
from ...utils.math import distance


class FaceModel(Model):
    def __init__(self, name, template_model, download_url, file_name):
        super().__init__(name, download_url, file_name)
        self.template_model = template_model

    @abstractmethod
    def detect_face(self, frame: numpy.ndarray):
        pass

    def get_eye_aspect_ratio(self, landmarks):
        blink_left_score = FaceModel.detect_blink(landmarks, self.template_model.get_left_eye_ear_indices())
        blink_right_score = FaceModel.detect_blink(landmarks, self.template_model.get_right_eye_ear_indices())

        return (blink_left_score + blink_right_score) / 2

    @staticmethod
    def detect_blink(landmarks, eye_indices_ear):
        p1 = landmarks[eye_indices_ear[0]]
        p2 = landmarks[eye_indices_ear[1]]
        p3 = landmarks[eye_indices_ear[2]]
        p4 = landmarks[eye_indices_ear[3]]
        p5 = landmarks[eye_indices_ear[4]]
        p6 = landmarks[eye_indices_ear[5]]

        vertical_distance_1 = distance(p2, p6)
        vertical_distance_2 = distance(p3, p5)

        horizontal_distance = distance(p1, p4)

        return (vertical_distance_1 + vertical_distance_2) / (2 * horizontal_distance)
