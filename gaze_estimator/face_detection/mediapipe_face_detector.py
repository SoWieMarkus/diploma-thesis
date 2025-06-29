from .face_detector import FaceDetector, FaceTemplateModel
from .face import FaceLandmark
import os
import numpy
import mediapipe

ASSETS_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "..", "data", "assets")
TEMPLATE_PATH = os.path.join(
    ASSETS_DIRECTORY,
    "template_model_3d_mediapipe.csv",
)


class MediapipeFaceTemplateModel(FaceTemplateModel):
    """Face template model for MediaPipe, containing 468 3D landmarks."""

    def __init__(self):
        template_path = os.path.abspath(TEMPLATE_PATH)

        super().__init__(template_path)
        if len(self.landmarks) != 468:
            raise ValueError(
                "Something went wrong while reading the 3D template model. "
                "It is supposed to contain 468 face landmarks."
            )

    def get_right_eye_ear_indices(self):
        return [33, 160, 158, 133, 153, 144]

    def get_left_eye_ear_indices(self):
        return [362, 385, 387, 263, 373, 380]

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


class MediapipeFaceDetector(FaceDetector):
    """Face detector using MediaPipe."""

    def __init__(self):
        template_model = MediapipeFaceTemplateModel()
        super().__init__("mediapipe", template_model)

        self.detector = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=1,  # max number of faces that are visible in the frame
            static_image_mode=True,  # ignore images before this frame in the detection
            min_detection_confidence=0.8,  # min confidence to detect a face in the frame
            min_tracking_confidence=0.8,  # min confidence to detect the facial landmarks
            refine_landmarks=True,  # refine the landmarks around the eyes and mouth
        )

    def detect_face(self, frame, meta):
        # Convert the BGR image to RGB
        # see https://stackoverflow.com/questions/39316447/opencv-giving-wrong-color-to-colored-images-on-loading
        frame_bgr = frame[:, :, ::-1]

        # Detect face landmarks using MediaPipe
        result = self.detector.process(frame_bgr)

        # Check if MediaPipe detected a face
        if not result.multi_face_landmarks:
            return None

        # We only check for one face in the frame, because on a bicycle there is usually only one person.
        # In the config of the detector we set `max_num_faces=1`, so there is always only one face detected.
        # So we take the first detected face landmarks.
        predictions = result.multi_face_landmarks[0]

        face_landmarks = []
        for prediction in predictions.landmark:
            # When refining the landmarks, mediapipe detects more than 468 landmarks.
            # This is a problem for us, since the template model only contains 468 landmarks.
            # So we stop processing the landmarks when we reach 468.
            if len(face_landmarks) >= 468:
                break

            image_x = prediction.x * meta.width
            image_y = prediction.y * meta.height

            face_landmark = FaceLandmark(
                normalized_x=prediction.x,
                normalized_y=prediction.y,
                normalized_z=prediction.z,
                image_x=image_x,
                image_y=image_y,
            )

            face_landmarks.append(face_landmark)
        return self.process_landmarks(face_landmarks, meta)
