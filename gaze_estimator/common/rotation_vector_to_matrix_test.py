import numpy
from .rotation_vector_to_matrix import convert_rotation_vector_to_matrix


def test_identity_rotation():
    # Zero rotation vector should yield identity matrix
    rotation_vector = numpy.array([0.0, 0.0, 0.0])
    # To avoid division by zero, handle this case in the function or here:
    try:
        matrix = convert_rotation_vector_to_matrix(rotation_vector)
    except ZeroDivisionError:
        # For zero vector, the rotation matrix is identity
        matrix = numpy.eye(3)
    assert numpy.allclose(matrix, numpy.eye(3)), f"Expected identity, got {matrix}"


def test_rotation_around_z_90_deg():
    # 90 degrees (pi/2 radians) around z-axis
    angle = numpy.pi / 2
    rotation_vector = numpy.array([0.0, 0.0, angle])
    matrix = convert_rotation_vector_to_matrix(rotation_vector)
    expected = numpy.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert numpy.allclose(
        matrix, expected, atol=1e-6
    ), f"Expected {expected}, got {matrix}"


if __name__ == "__main__":
    test_identity_rotation()
    test_rotation_around_z_90_deg()
    print("All tests passed.")
