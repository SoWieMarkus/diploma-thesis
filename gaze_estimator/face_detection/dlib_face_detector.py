from .face_detector import FaceDetector, FaceTemplateModel
from .face import FaceLandmark
import os
import numpy
import dlib

ASSETS_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data", "assets")

MODEL_FILE_NAME = "face-model-dlib.dat"
MODEL_FILE_PATH = os.path.join(ASSETS_DIRECTORY, MODEL_FILE_NAME)

TEMPLATE_PATH = os.path.join(
    ASSETS_DIRECTORY,
    "template-model-3d-dlib.csv",
)


class DlibFaceTemplateModel(FaceTemplateModel):
    """
    Face template model for Dlib, containing 68 3D landmarks.
    """

    def __init__(self):
        template_path = os.path.abspath(TEMPLATE_PATH)
        super().__init__(template_path)
        if len(self.landmarks) != 68:
            raise ValueError(
                "Something went wrong while reading the 3D template model. "
                "It is supposed to contain 468 face landmarks."
            )

    def get_right_eye_ear_indices(self):
        return [36, 37, 38, 39, 40, 41]

    def get_left_eye_ear_indices(self):
        return [42, 43, 44, 45, 46, 47]

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


class DlibFaceDetector(FaceDetector):
    """Face detector using Dlib."""

    def __init__(self):
        template_model = DlibFaceTemplateModel()
        super().__init__("dlib", template_model)
        model_path = os.path.abspath(MODEL_FILE_PATH)
        self.frontal_face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(str(model_path))

    def detect_face(self, frame, meta):

        # Detect faces in the frame using Dlib's frontal face detector
        bounding_boxes_of_faces = self.frontal_face_detector(frame, 0)

        # Check if a face was detected
        if len(bounding_boxes_of_faces) == 0:
            return None

        # Since there usally is only one face in the frame while riding a bike, we take the first detected face
        bounding_box_of_face = bounding_boxes_of_faces[0]

        # Predict the facial landmarks using Dlib's shape predictor
        predictions = self.shape_predictor(frame, bounding_box_of_face)

        face_landmarks = []
        for prediction in predictions.parts():
            face_landmark = FaceLandmark(
                normalized_x=prediction.x / meta.width,
                normalized_y=prediction.y / meta.height,
                normalized_z=1.0,  # Dlib does not provide z-coordinates
                image_x=prediction.x,
                image_y=prediction.y,
            )
            face_landmarks.append(face_landmark)
        return self.process_landmarks(face_landmarks, meta)
