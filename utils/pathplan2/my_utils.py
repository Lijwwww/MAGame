import math


def path_plan2_coordinate_world2map(x, y):
    x1 = x * 10 + 100
    y1 = (-y) * 10 + 50
    return x1, y1  

def path_plan2_coordinate_map2world(x, y):
    x1 = (x - 100) / 10
    y1 = (50 - y) / 10
    return x1, y1

def find_nearest_point_on_path(x, y, path):
    min_distance = float('inf'); min_p = len(path) - 1
    for p, point in enumerate(path):
        distance = math.sqrt((point.x - x)**2 + (point.y - y)**2)
        if distance < min_distance:
            min_distance = distance; min_p = p
    return min_p