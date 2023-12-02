import fractions
import os

import cv2
import numpy


class Video:

    def __init__(self, file) -> None:
        self.file = file
        self.name = os.path.basename(file)

        self.video_capture = cv2.VideoCapture(file)

        self.frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.timestamp_first_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))

        center = (self.width / 2, self.height / 2)

        self.camera_matrix = numpy.array(
            [
                [self.width, 0.0, center[0]],
                [0.0, self.width, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype="double",
        )

        # todo 5
        self.distortion_coefficients = numpy.zeros((4, 1))

        aspect_ratio = fractions.Fraction(int(self.width), int(self.height))

        simplified_ratio = aspect_ratio.limit_denominator()

        self.aspect_ratio = {
            "x": simplified_ratio.numerator,
            "y": simplified_ratio.denominator,
        }

    def get_timestamp_of_frame(self, frame_number):
        return self.timestamp_first_frame + frame_number

    def __str__(self):
        return f"{self.file} ->[{self.width}px | {self.height}px], {self.aspect_ratio['x']}:{self.aspect_ratio['y']}, {self.frame_rate} fps, {self.total_frames} frames"
