from .bounding_box import get_bounding_box


def test_bounding_box_basic():
    points = [(1, 2), (3, 4), (5, 0), (2, 8)]
    bbox = get_bounding_box(points)
    assert bbox == ((1, 0), (5, 8))


def test_bounding_box_single_point():
    points = [(7, 7)]
    bbox = get_bounding_box(points)
    assert bbox == ((7, 7), (7, 7))


def test_bounding_box_negative_coords():
    points = [(-1, -2), (3, 4), (0, 0)]
    bbox = get_bounding_box(points)
    assert bbox == ((-1, -2), (3, 4))


def test_bounding_box_horizontal_line():
    points = [(1, 5), (3, 5), (2, 5)]
    bbox = get_bounding_box(points)
    assert bbox == ((1, 5), (3, 5))


def test_bounding_box_vertical_line():
    points = [(4, 1), (4, 3), (4, 2)]
    bbox = get_bounding_box(points)
    assert bbox == ((4, 1), (4, 3))
