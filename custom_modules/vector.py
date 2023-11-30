import math


class Vector:
    """Vector class to help with linalgebra operations."""

    def __init__(self, *coordinates):
        self.coordinates = coordinates

    def __add__(self, other):
        return Vector(*(x + y for x, y in zip(self.coordinates, other.coordinates)))

    def __sub__(self, other):
        return Vector(*(x - y for x, y in zip(self.coordinates, other.coordinates)))

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot(other)
        else:
            return Vector(*(x * other for x in self.coordinates))

    def __truediv__(self, scalar):
        return Vector(*(x / scalar for x in self.coordinates))

    def dot(self, other):
        return sum(x * y for x, y in zip(self.coordinates, other.coordinates))

    def norm(self):
        return math.sqrt(self.dot(self))

    def distance(self, other):
        return (self - other).norm()

    def normalize(self):
        return self / self.norm()

    def __getitem__(self, index):
        return self.coordinates[index]

