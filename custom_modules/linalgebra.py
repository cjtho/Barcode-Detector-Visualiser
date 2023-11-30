import math
from .vector import Vector


def calculate_mean_point(points):
    total = len(points)
    mean_x = sum(point[0] for point in points) / total
    mean_y = sum(point[1] for point in points) / total
    return mean_x, mean_y


def translate_line(line, translate_vector):
    segment_start, segment_end = line
    new_start = (round(segment_start[0] + translate_vector[0]),
                 round(segment_start[1] + translate_vector[1]))
    new_end = (round(segment_end[0] + translate_vector[0]),
               round(segment_end[1] + translate_vector[1]))
    return new_start, new_end


def min_distance_to_line(line, points):
    line_start, line_end = [Vector(*point) for point in line]
    line_vector = line_end - line_start

    minimum_distance = float("inf")
    closest_point = None
    closest_point_projection = None
    for point in points:
        point_vector = Vector(*point) - line_start
        projection_length = (point_vector * line_vector) / (line_vector.norm() ** 2)
        projection_vector = line_vector * projection_length
        distance_to_line = point_vector.distance(projection_vector)
        if distance_to_line < minimum_distance:
            minimum_distance = distance_to_line
            closest_point = point
            closest_point_projection = (projection_vector + line_start)

    return minimum_distance, closest_point, closest_point_projection.coordinates


def rotate_points(points, theta, mean=(0, 0), angle_mode="deg"):
    rotated_points = []
    mean_r, mean_c = mean
    theta = math.radians(theta) if angle_mode == "deg" else theta
    for r, c in points:
        r_rotated = mean_r + (r - mean_r) * math.cos(theta) - (c - mean_c) * math.sin(theta)
        c_rotated = mean_c + (r - mean_r) * math.sin(theta) + (c - mean_c) * math.cos(theta)
        rotated_points.append((r_rotated, c_rotated))

    return rotated_points


def calculate_radius(points):
    mean_r, mean_c = calculate_mean_point(points)
    farthest_point = max(points, key=lambda p: math.dist((mean_r, mean_c), p))
    radius = math.dist((mean_r, mean_c), farthest_point)
    return radius
