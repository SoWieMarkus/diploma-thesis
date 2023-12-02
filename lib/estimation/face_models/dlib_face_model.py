import bz2

import dlib

from .face_model import FaceModel
from .templates.dlib_template_model import DlibFaceTemplateModel
from ..face.face import Face, FaceLandmark


class DlibFaceModel(FaceModel):
    """
    Class that allows face and facial landmarks recognition with dlib.
    """

    def __init__(self):
        super().__init__(
            name="dlib",
            template_model=DlibFaceTemplateModel(),
            # Download Shape Predictor of dlib
            download_url="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            file_name="shape_predictor.dat.bz2",
        )
        self.model_file = "shape_predictor_68_face_landmarks.dat"
        self.frontal_face_detector = None
        self.shape_predictor = None

    def get_extracted_model_path(self):
        return self.model_directory / self.model_file

    def get_zip_path(self):
        return self.model_directory / self.file_name

    def download(self):
        super().download()
        model_path = self.get_extracted_model_path()

        # extract shape predictor
        with bz2.BZ2File(self.get_zip_path(), "rb") as zip_file, open(
                model_path, "wb"
        ) as model_file:
            data = zip_file.read()
            model_file.write(data)

        # initialize the frontal face detector and shape predictor after download
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(str(model_path))

    def detect_face(self, frame):
        """
        Detect a face and facial landmarks with dlib based on a given frame of the video
        """
        height = frame.shape[0]
        width = frame.shape[1]

        # detect faces
        bounding_boxes_of_faces = self.frontal_face_detector(frame, 0)

        # check if a face was detected
        if len(bounding_boxes_of_faces) == 0:
            return None

        # choose the first detected face
        bounding_boxes_of_face = bounding_boxes_of_faces[0]

        # predict the facial landmarks
        predictions = self.shape_predictor(frame, bounding_boxes_of_face)

        # calculate ear score
        average_blink_score = self.get_eye_aspect_ratio(predictions.parts())

        landmarks = []
        for landmark in predictions.parts():
            #normalize width and height to a value between 0 and 1 like mediapipe does it
            x = landmark.x / width
            y = landmark.y / height
            # cant give any information about the depth yet. the normalized depth value of dlib will always be 1
            z = 1

            image_x = landmark.x
            image_y = landmark.y

            landmarks.append(FaceLandmark(x, y, z, image_x, image_y))

        return Face(landmarks, average_blink_score)
