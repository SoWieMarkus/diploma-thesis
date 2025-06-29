import numpy


def convert_rotation_vector_to_matrix(rotation_vector: numpy.array):
    # magnitude of vector represents angle in radians
    angle_of_rotation = numpy.linalg.norm(rotation_vector)
    if angle_of_rotation == 0:
        return numpy.eye(3)

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
