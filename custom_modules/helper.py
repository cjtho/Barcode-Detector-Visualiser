"""Code dump of useful functions used throughout. Not great practice but ¯\_(ツ)_/¯"""


def shape(mat):
    return len(mat), len(mat[0])


def deepcopy(mat):
    if isinstance(mat, list):
        return list(deepcopy(element) for element in mat)
    elif isinstance(mat, tuple):
        return tuple(deepcopy(element) for element in mat)
    else:
        return mat


def create_matrix(width, height, initial_value=None):
    return [[initial_value or 0.0 for _ in range(width)] for _ in range(height)]


def linspace(start, end, iterations):
    step = (end - start) / iterations
    curr_angle = start
    for _ in range(iterations):
        yield curr_angle
        curr_angle += step


def pairwise(iterable, end_points=False):
    n = len(iterable)
    for i in range(n if end_points else n - 1):
        yield iterable[i], iterable[(i + 1) % n]
