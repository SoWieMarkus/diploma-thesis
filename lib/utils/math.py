import numpy

import math

vector_to_string = lambda vector: f"{round(vector[0], 5)},{round(vector[1], 5)},{round(vector[2], 5)}"

distance = lambda p1, p2: math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y


def vector_from_string(string):
    parts = string.split(",")
    return numpy.array([float(parts[0]), float(parts[1]), float(parts[2])])


def convert_rotation_vector_to_matrix(rotation_vector):
    # magnitude of vector represents angle in radians
    angle_of_rotation = numpy.linalg.norm(rotation_vector)
    rotation_axis = rotation_vector / angle_of_rotation

    cosinus = numpy.cos(angle_of_rotation)
    sinus = numpy.sin(angle_of_rotation)

    t = 1 - cosinus

    x = rotation_axis[0]
    y = rotation_axis[1]
    z = rotation_axis[2]

    x1 = t * x * x + cosinus
    x2 = t * x * y - z * sinus
    x3 = t * x * z + y * sinus

    y1 = t * x * y + z * sinus
    y2 = t * y * y + cosinus
    y3 = t * y * z - x * sinus

    z1 = t * x * z - y * sinus
    z2 = t * y * z + x * sinus
    z3 = t * z * z + cosinus

    return numpy.array([[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]])


def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = numpy.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = numpy.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = numpy.arctan2(-rotation_matrix[2, 0], sy)
        z = numpy.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = numpy.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = numpy.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    return numpy.array([x, y, z])
