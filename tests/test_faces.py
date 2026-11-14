import cv2
import numpy as np
import pytest

from gaze_estimator.core.camera import Camera
from gaze_estimator.core.paths import ASSETS_DIR
from gaze_estimator.faces.base import FaceDetector, FaceTemplate, eye_aspect_ratio
from gaze_estimator.gaze.head_rotation import HeadRotationEstimator

MEDIAPIPE_INDICES = dict(
    right_eye=(33, 133),
    left_eye=(362, 263),
    mouth=(78, 308),
    nose=(240, 460),
    nose_tip=1,
    right_eye_ear=(33, 160, 158, 133, 153, 144),
    left_eye_ear=(362, 385, 387, 263, 373, 380),
)


class TemplateOnlyDetector(FaceDetector):
    """Detector without a model: the pose estimation is tested on projected template points."""

    name = "template"

    def __init__(self):
        self.template = FaceTemplate.from_csv(
            ASSETS_DIR / "template-model-3d-mediapipe.csv", 468, **MEDIAPIPE_INDICES
        )

    def detect_landmarks(self, frame_bgr):
        return None


def test_pose_estimation_recovers_a_known_pose():
    detector = TemplateOnlyDetector()
    camera = Camera.default(640, 480)
    rvec = np.array([0.1, -0.2, 0.05])
    tvec = np.array([0.03, -0.02, 0.5])
    points, _ = cv2.projectPoints(
        detector.template.landmarks, rvec, tvec, camera.matrix, camera.distortion_coefficients
    )

    face = detector.estimate_pose(points.reshape(-1, 2), camera)
    np.testing.assert_allclose(face.rotation_vector, rvec, atol=1e-4)
    np.testing.assert_allclose(face.translation_vector, tvec, atol=1e-4)
    # The face center (eyes + nose) lies a few centimeters away from the template origin.
    assert face.distance == pytest.approx(0.5, abs=0.05)
    # The normalizing rotation is orthonormal and its z-axis points at the face center.
    np.testing.assert_allclose(
        face.normalizing_rotation @ face.normalizing_rotation.T, np.eye(3), atol=1e-9
    )
    np.testing.assert_allclose(face.normalizing_rotation[2], face.center / face.distance, atol=1e-9)

    # A frontal face (no rotation) "looks" straight at the camera with the head rotation baseline.
    frontal, _ = cv2.projectPoints(
        detector.template.landmarks, np.zeros(3), tvec, camera.matrix, np.zeros(4)
    )
    face = detector.estimate_pose(frontal.reshape(-1, 2), camera)
    gaze = HeadRotationEstimator().estimate(None, face, camera)
    np.testing.assert_allclose(gaze.vector, [0, 0, -1], atol=1e-4)


def test_eye_aspect_ratio():
    open_eye = np.array([[0, 0], [1, 1], [2, 1], [3, 0], [2, -1], [1, -1]], dtype=float)
    assert eye_aspect_ratio(open_eye, (0, 1, 2, 3, 4, 5)) == pytest.approx(2 / 3)


def test_dlib_template():
    pytest.importorskip("dlib")
    from gaze_estimator.faces.dlib import dlib_template

    assert dlib_template().landmarks.shape == (68, 3)
