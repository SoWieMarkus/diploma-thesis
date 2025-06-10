import numpy
from .vector_from_pitch_yaw import get_vector_from_pitch_and_yaw


def test_zero_angles():
    vec = get_vector_from_pitch_and_yaw(0.0, 0.0)
    expected = -numpy.array([0.0, 0.0, 1.0])
    numpy.testing.assert_allclose(vec, expected, atol=1e-7)


def test_pitch_only():
    pitch = numpy.pi / 2
    vec = get_vector_from_pitch_and_yaw(pitch, 0.0)
    expected = -numpy.array([0.0, 1.0, 0.0])
    numpy.testing.assert_allclose(vec, expected, atol=1e-7)


def test_yaw_only():
    yaw = numpy.pi / 2
    vec = get_vector_from_pitch_and_yaw(0.0, yaw)
    expected = -numpy.array([1.0, 0.0, 0.0])
    numpy.testing.assert_allclose(vec, expected, atol=1e-7)
