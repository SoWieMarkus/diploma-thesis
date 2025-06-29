import numpy


def get_vector_from_pitch_and_yaw(pitch: float, yaw: float) -> numpy.ndarray:
    """
    Convert pitch and yaw angles to a 3D gaze vector.
    Args:
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
    Returns:
        numpy.ndarray: A 3D gaze vector.
    """
    return -numpy.array(
        [
            numpy.cos(pitch) * numpy.sin(yaw),
            numpy.sin(pitch),
            numpy.cos(pitch) * numpy.cos(yaw),
        ]
    )
