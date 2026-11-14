import math

import numpy as np
import pytest

from gaze_estimator.core.camera import Camera
from gaze_estimator.core.geometry import (
    Display,
    intersect_camera_plane,
    pitch_yaw_from_vector,
    vector_from_pitch_yaw,
)


def test_zero_pitch_yaw_looks_at_camera():
    np.testing.assert_allclose(vector_from_pitch_yaw(0, 0), [0, 0, -1], atol=1e-12)


def test_pitch_yaw_roundtrip():
    for pitch, yaw in [(0.1, -0.3), (-0.4, 0.7), (0.0, 1.2)]:
        assert pitch_yaw_from_vector(vector_from_pitch_yaw(pitch, yaw)) == pytest.approx(
            (pitch, yaw)
        )


def test_intersection_single_and_batch():
    origin = np.array([0.1, -0.2, 0.5])
    direction = np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(intersect_camera_plane(origin, direction), [0.1, -0.2, 0.0])

    origins = np.array([[0, 0, 0.5], [0, 0, 0.5]])
    directions = np.array([[0, 0.5, -0.5], [0, 0, 1.0]])  # second one looks away
    points = intersect_camera_plane(origins, directions)
    np.testing.assert_allclose(points[0], [0, 0.5, 0])
    assert np.isnan(points[1]).all()


def test_display_bounds_follow_the_thesis_definition():
    # Square display l with offset p: x in [-l/2, l/2], y in [-p, l - p]
    display = Display.square(0.2, 0.05)
    assert display.bounds == pytest.approx((-0.1, 0.1, -0.05, 0.15))
    points = np.array([[0, 0.1, 0], [0, -0.06, 0], [0.11, 0, 0], [np.nan, np.nan, np.nan]])
    assert display.contains(points).tolist() == [True, False, False, False]


def test_angular_distance():
    display = Display(0.1, 0.1, 0.0)  # x in [-0.05, 0.05], y in [0, 0.1]
    center = np.array([[0.0, 0.0, 0.5]])
    hit = display.angular_distance(center, np.array([[0.0, 0.1, -0.5]]))
    assert hit[0] == 0.0

    # Looking 0.1 m to the right of the right edge at the height of the camera.
    direction = np.array([[0.15, 0.0, -0.5]])
    expected = math.atan2(0.15, 0.5) - math.atan2(0.05, 0.5)
    assert display.angular_distance(center, direction)[0] == pytest.approx(expected, abs=1e-3)


def test_camera_profiles():
    default = Camera.default(1920, 1080)
    assert default.fx == 1920 and default.cx == 960
    assert Camera.from_hfov(1920, 1080, default.hfov_deg).fx == pytest.approx(1920)

    scaled = Camera.from_profile(
        {"width": 1920, "height": 1080, "fx": 1500, "fy": 1500, "cx": 960, "cy": 540}, 960, 540
    )
    assert scaled.fx == 750 and scaled.cy == 270
    assert Camera.from_dict(scaled.to_dict()) == scaled
