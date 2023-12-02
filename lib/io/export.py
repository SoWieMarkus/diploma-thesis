import pathlib

import cv2
import numpy

from .frame_face_data import FrameFaceData
from .frame_gaze_data import GazeFrameData
from .frame_meta_data import FrameMetaData

SCRIPT_PATH = pathlib.Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_PATH / "../.." / "output"


class MetaDataExporter:

    def __init__(self, parent_directory):
        self.directory = parent_directory
        self.file_name = "base.meta"
        self.frames_data = []
        self.images_directory = self.directory / "images"
        self.images_directory.mkdir(parents=True, exist_ok=True)

    def save_image(self, image, frame_number, timestamp):
        self.frames_data.append(FrameMetaData(frame_number=frame_number, timestamp=timestamp, frame=image))

        file_name = f"{frame_number}.jpg"
        cv2.imwrite(
            str(self.images_directory / file_name),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

    def get_output_file(self):
        return self.directory / self.file_name

    def save_to_file(self, completed: bool):
        content = [f"{completed}"]
        for data_frame in self.frames_data:
            content.append(data_frame.export())
        string_content = "\n\n".join(content)

        with open(self.get_output_file(), "w") as file:
            file.write(string_content)

    def read(self):
        path = self.get_output_file()

        if not path.is_file():
            return

        with open(path, "r") as file:
            content = file.read()
            parts = content.split("\n\n")
            self.completed = bool(parts[0])

            for index in range(1, len(parts)):
                self.frames_data.append(FrameMetaData.from_string(parts[index]))


class FaceExporter:
    def __init__(self, parent_directory, face_model) -> None:
        self.face_model = face_model
        self.directory = parent_directory / "face" / face_model.name
        self.face_model_name = face_model.name
        self.image_directory = self.directory / "images"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.image_directory.mkdir(parents=True, exist_ok=True)
        self.frames_data = []
        self.completed = False

    def set_completed(self, completed):
        self.completed = completed

    def draw_landmarks(self, frame, points):
        count = 0

        height, width = frame.shape[:2]

        # Double the size of the image
        new_width = width * 2
        new_height = height * 2

        # Resize the image to the new dimensions
        # frame = cv2.resize(frame, (new_width, new_height))

        for point in points:
            color = (0, 0, 255)
            size = 2

            found = False
            if numpy.isin(
                    count, self.face_model.template_model.get_right_eye_indices()
            ):
                color = (0, 255, 255)
                size = 3
                found = True
            if numpy.isin(count, self.face_model.template_model.get_left_eye_indices()):
                color = (255, 0, 255)
                size = 3
                found = True
            if numpy.isin(count, self.face_model.template_model.get_mouth_indices()):
                color = (255, 0, 0)
                size = 3
                found = True
            if numpy.isin(count, self.face_model.template_model.get_nose_indices()):
                color = (0, 255, 0)
                size = 3
                found = True
            if count == self.face_model.template_model.get_nose_tip_index():
                color = (0, 255, 0)
                size = 3
                found = True
            if count == self.face_model.template_model.get_chin_index():
                color = (255, 255, 255)
                size = 3
                found = True

            # point = (point[0] * 2, point[1]*2)
            pt = numpy.round(point).astype(numpy.int32).tolist()
            cv2.circle(frame, pt, size, color, cv2.FILLED)

            # cv2.putText(frame, str(count), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            count += 1
        return frame

    def save_face(self, face, frame, frame_number, timestamp):
        self.save_frame_with_landmarks(
            face=face, frame=frame, frame_number=frame_number
        )

        frame = FrameFaceData(frame_number, face, timestamp)
        self.frames_data.append(frame)

    def save_frame_with_landmarks(self, face, frame, frame_number):
        frame = self.draw_landmarks(frame=frame, points=face.image_points)
        image_name = f"{frame_number}.jpg"
        file_path = self.image_directory / image_name

        padding = 50

        # Calculate new coordinates for the bounding box with padding

        (x1, y1), (x2, y2) = face.bounding_box

        x1_pad = max(0, (x1) - padding)
        y1_pad = max(0, (y1) - padding)

        x2_pad = min(frame.shape[1], (x2) + padding)
        y2_pad = min(frame.shape[0], (y2) + padding)

        cropped_image = frame[int(y1_pad): int(y2_pad), int(x1_pad): int(x2_pad)]

        if not cv2.imwrite(str(file_path), cropped_image):
            raise Exception("Could not write image")

    def get_output_file(self):
        return self.directory / f"{self.face_model_name}.face"

    def save_to_file(self):
        content = [f"${self.completed}"]
        for data_frame in self.frames_data:
            content.append(data_frame.export())
        string_content = "\n\n".join(content)

        with open(self.get_output_file(), "w") as file:
            file.write(string_content)

    def get_by_frame_number(self, frame_number: int):
        for frame in self.frames_data:
            if frame.frame_number == frame_number:
                return frame
        return None

    def read(self):
        path = self.get_output_file()

        if not path.is_file():
            return

        with open(path, "r") as file:
            content = file.read()
            parts = content.split("\n\n")
            self.completed = bool(parts[0])

            for index in range(1, len(parts)):
                self.frames_data.append(FrameFaceData.from_string(parts[index]))


class GazeExporter:
    def __init__(self, parent_directory, gaze_model_name: str) -> None:
        self.directory = parent_directory / "gaze"
        self.gaze_model_name = gaze_model_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gaze_data = []
        self.completed = False

    def save_gaze(self, gaze_frame_data):
        self.gaze_data.append(gaze_frame_data)

    def set_completed(self, completed):
        self.completed = completed

    def get_output_file(self):
        return self.directory / f"{self.gaze_model_name}.gaze"

    def save_to_file(self):
        content = [f"${self.completed}"]
        for data_frame in self.gaze_data:
            content.append(data_frame.export())
        string_content = "\n\n".join(content)

        with open(self.get_output_file(), "w") as file:
            file.write(string_content)


class Exporter:
    def __init__(self, video_name: str) -> None:
        self.name = str(video_name).split(".")[0]

        self.directory = OUTPUT_PATH / self.name
        self.directory.mkdir(parents=True, exist_ok=True)

        self.face_exporter = {}
        self.gaze_exporter = {}

        self.meta_exporter = MetaDataExporter(parent_directory=self.directory)

    def register(self, model_manager):
        for face_model in model_manager.face_models:
            self.face_exporter[face_model.name] = FaceExporter(
                parent_directory=self.directory, face_model=face_model
            )

        for gaze_model in model_manager.gaze_models:
            for face_model in model_manager.face_models:
                name = self.get_gaze_exporter_name(
                    gaze_model_name=gaze_model.name, face_model_name=face_model.name
                )
                self.gaze_exporter[name] = GazeExporter(
                    parent_directory=self.directory, gaze_model_name=name
                )

    def get_gaze_exporter_name(self, gaze_model_name, face_model_name):
        return f"{face_model_name}_{gaze_model_name}"

    def save_frame(self, frame, frame_number, timestamp):
        self.meta_exporter.save_image(frame, frame_number, timestamp)

    def save_face(self, face_model_name, face, frame, frame_number, timestamp):
        exporter = self.face_exporter[face_model_name]
        if exporter is None:
            raise ValueError(
                f"Face model with the name {face_model_name} is not registered."
            )

        exporter.save_face(
            face=face, frame=frame, frame_number=frame_number, timestamp=timestamp
        )

    def save_gaze(
            self,
            gaze_model_name: str,
            face_model_name: str,
            frame_number,
            timestamp,
            pitch,
            yaw,
            vector,
            intersection,
            hit
    ):
        name = self.get_gaze_exporter_name(
            gaze_model_name=gaze_model_name, face_model_name=face_model_name
        )

        exporter = self.gaze_exporter[name]
        if exporter is None:
            raise ValueError(f"Gaze model with the name {name} is not registered.")

        exporter.save_gaze(
            GazeFrameData(
                frame_number=frame_number,
                timestamp=timestamp,
                pitch=pitch,
                yaw=yaw,
                vector=vector,
                intersection=intersection,
                hit=hit
            )
        )

    def save_meta_data(self, completed: bool):
        self.meta_exporter.save_to_file(completed)

    def save_gaze_data(self, completed: bool):
        for gaze_exporter in self.gaze_exporter.values():
            gaze_exporter.set_completed(completed)
            gaze_exporter.save_to_file()

    def save_face_data(self, completed: bool):
        for face_exporter in self.face_exporter.values():
            face_exporter.set_completed(completed)
            face_exporter.save_to_file()

    def read_face_data(self):
        for face_exporter in self.face_exporter.values():
            face_exporter.read()
