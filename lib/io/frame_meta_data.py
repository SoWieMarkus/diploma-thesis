import cv2
import numpy


class FrameMetaData:

    def __init__(self, frame_number, timestamp, frame):
        self.frame_number = frame_number
        self.timestamp = timestamp

        if frame is None:
            self.sharpness = None
            self.contrast = None
            self.brightness = None
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        self.sharpness = laplacian.var()
        self.contrast = gray.std()
        self.brightness = (numpy.mean(gray) / 255) * 100

    def export(self):
        return "\n".join(
            [str(self.frame_number),
             str(self.timestamp),
             str(self.brightness),
             str(self.contrast),
             str(self.sharpness)]
        )

    @staticmethod
    def from_string(content):
        parts = content.split("\n")
        frame_number = int(parts[0])
        timestamp = int(parts[1])

        frame_data = FrameMetaData(frame_number=frame_number, timestamp=timestamp, frame=None)
        frame_data.brightness = float(parts[2])
        frame_data.contrast = float(parts[3])
        frame_data.sharpness = float(parts[4])
        return frame_data
