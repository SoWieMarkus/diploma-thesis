from .face_detector import FaceDetector, FaceTemplateModel
from .face import FaceLandmark
import os
import numpy
import bz2
import torch
import logging
import dlib

ASSETS_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data", "assets")

MODEL_DOWNLOAD_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

MODEL_ZIP_FILE_NAME = "shape_predictor_68_face_landmarks.dat.bz2"
MODEL_ZIP_FILE_PATH = os.path.join(ASSETS_DIRECTORY, MODEL_ZIP_FILE_NAME)

MODEL_FILE_NAME = "shape_predictor_68_face_landmarks.dat"
MODEL_FILE_PATH = os.path.join(ASSETS_DIRECTORY, MODEL_FILE_NAME)

TEMPLATE_PATH = os.path.join(
    ASSETS_DIRECTORY,
    "template_model_3d_dlib.csv",
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
        model_zip_path = os.path.abspath(MODEL_ZIP_FILE_PATH)
        model_path = os.path.abspath(MODEL_FILE_PATH)
        if not os.path.exists(model_path):
            logging.info("Downloading Dlib face model...")

            # Download shape predictor (68 landmarks) as .bz2 file
            torch.hub.download_url_to_file(
                MODEL_DOWNLOAD_URL, model_zip_path, progress=True
            )

            # Extract shape predictor from the .bz2 file
            with bz2.BZ2File(model_zip_path, "rb") as zip_file, open(
                model_path, "wb"
            ) as model_file:
                data = zip_file.read()
                model_file.write(data)

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
