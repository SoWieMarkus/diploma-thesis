import cv2
import numpy
from ..face_detection import Face, FaceTemplateModel
from ..common.camera import VideoCaptureMetaData

DEFAULT_STYLE = ((0, 0, 255), 2)


def draw_landmarks(
    frame: numpy.ndarray,
    face: Face,
    template_model: FaceTemplateModel,
    only_draw_important: bool = False,
) -> numpy.ndarray:
    """
    Draws the landmarks of a face on the given frame.
    Args:
        frame (numpy.ndarray): The image frame on which to draw the landmarks.
        face (Face): The face object containing the landmarks.
        template_model (FaceTemplateModel): The template model used for landmark indices.
        only_draw_important (bool): If True, only draws important landmarks.
    Returns:
        numpy.ndarray: The frame with the landmarks drawn on it.
    """
    style_map = {}

    def add_indices(indices, color, size):
        for idx in numpy.atleast_1d(indices):
            style_map[int(idx)] = (color, size)

    # Add special highlights to important landmarks (color and size)
    add_indices(template_model.get_right_eye_indices(), (0, 255, 255), 3)
    add_indices(template_model.get_left_eye_indices(), (255, 0, 255), 3)
    add_indices(template_model.get_mouth_indices(), (255, 0, 0), 3)
    add_indices(template_model.get_nose_indices(), (0, 255, 0), 3)
    add_indices(template_model.get_nose_tip_index(), (0, 255, 0), 3)

    indices_to_draw = range(len(face.image_points))
    if only_draw_important:
        indices_to_draw = template_model.get_important_indices()

    for idx in indices_to_draw:
        image_point = face.image_points[idx]
        pt = numpy.round(image_point).astype(numpy.int32).tolist()
        color, size = style_map.get(idx, DEFAULT_STYLE)
        cv2.circle(frame, pt, size, color, cv2.FILLED)

    return frame
