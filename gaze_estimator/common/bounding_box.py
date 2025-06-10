def get_bounding_box(
    image_points: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Calculate the bounding box of a set of image points.
    The bounding box is defined by the minimum and maximum x and y coordinates of the points.

    Args:
        image_points (list[tuple[int, int]]): List of image points in the format [(x1, y1), (x2, y2), ...].

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: Bounding box in the format ((x_min, y_min), (x_max, y_max)).
    """
    min_x = min(point[0] for point in image_points)
    max_x = max(point[0] for point in image_points)
    min_y = min(point[1] for point in image_points)
    max_y = max(point[1] for point in image_points)

    return ((min_x, min_y), (max_x, max_y))
