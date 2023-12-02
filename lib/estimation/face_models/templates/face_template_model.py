import csv
import logging
import pathlib
from abc import ABC, abstractmethod

import numpy

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent
MODELS_PATH = SCRIPT_PATH / "../../../.." / "models"


class FaceTemplateModel(ABC):
    def __init__(self, file):
        template_3d_model_path = MODELS_PATH / file

        logging.info(f"Reading 3D face template model '{str(template_3d_model_path)}'")

        with open(MODELS_PATH / template_3d_model_path, newline="\n") as csv_file:
            reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            points = []
            for row in reader:
                numbers = []
                for column in row:
                    numbers.append(float(column))
                points.append(numbers)
            self.landmarks = numpy.array(points, dtype="double")

    @abstractmethod
    def get_right_eye_ear_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_left_eye_ear_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_right_eye_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_left_eye_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_mouth_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_nose_indices(self):
        raise NotImplementedError()

    @abstractmethod
    def get_nose_tip_index(self):
        raise NotImplementedError()

    @abstractmethod
    def get_chin_index(self):
        raise NotImplementedError()
