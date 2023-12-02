import mediapipe
import numpy

from .face_model import FaceModel
from .templates.mediapipe_template_model import MediapipeFaceTemplateModel
from ..face.face import FaceLandmark, Face
from ...utils.math import Point


class MediapipeFaceModel(FaceModel):
    """
    Class to detect a face and facial landmarks with Mediapipe
    """

    def __init__(self):
        super().__init__(
            name="mediapipe",
            template_model=MediapipeFaceTemplateModel(),
            download_url=None,
            file_name=None,
        )

        self.detector = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=1,  # max number of faces that are visible in the frame
            static_image_mode=True,  # ignore images before this frame in the detection
            min_detection_confidence=0.8,  # min confidence to detect a face in the frame
            min_tracking_confidence=0.8,  # min confidence to detect the facial landmarks
            refine_landmarks=True  # refine the landmarks around the eyes and mouth. note this will add an additional 10 landmarks that will be discarded later
        )

    def detect_face(self, frame: numpy.ndarray):
        height = frame.shape[0]
        width = frame.shape[1]

        # Convert the BGR image to RGB
        # see https://stackoverflow.com/questions/39316447/opencv-giving-wrong-color-to-colored-images-on-loading
        frame_bgr = frame[:, :, ::-1]

        result = self.detector.process(frame_bgr)

        # check if mediapipe detected a face
        if not result.multi_face_landmarks:
            return None

        face_landmarks = result.multi_face_landmarks[0]
        landmarks = []

        image_points = []
        for landmark in face_landmarks.landmark:
            # calculate pixel coordinates for the normalized landmarks for ear calculation
            x = landmark.x
            image_x = x * width
            y = landmark.y
            image_y = y * height

            image_points.append(Point(image_x, image_y))

        # calculate ear
        average_blink_score = self.get_eye_aspect_ratio(image_points)

        count = 0
        for landmark in face_landmarks.landmark:
            if count == 468:
                break
            count += 1
            x = landmark.x
            #calculate image coordinates in pixel for the normalized landmark position
            image_x = x * width

            y = landmark.y
            image_y = y * height

            z = landmark.z
            landmarks.append(FaceLandmark(x, y, z, image_x, image_y))

        return Face(landmarks, average_blink_score)
