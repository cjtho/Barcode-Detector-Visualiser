from .helper import shape

class BorderHandler:
    """Pre-processes the borders in the convolution."""

    def __init__(self, mode, kernel):
        self.mode = mode
        self.kernel = kernel
        self.funcs = {
            "border_repeat": self._handle_border_repeat,
            "border_ignore_zeros": self._handle_border_ignore_zeros,
        }
        self.func = self.funcs[self.mode]

    def get_border_coordinates(self, rows, cols):
        top_border = ((i, j) for i in range(self.kernel.half_height)
                      for j in range(cols))
        bottom_border = ((i, j) for i in range(rows - self.kernel.half_height, rows)
                         for j in range(cols))
        left_border = ((i, j) for i in range(self.kernel.half_height, rows - self.kernel.half_height)
                       for j in range(self.kernel.half_width))
        right_border = ((i, j) for i in range(self.kernel.half_height, rows - self.kernel.half_height)
                        for j in range(cols - self.kernel.half_width, cols))

        yield from top_border
        yield from bottom_border
        yield from left_border
        yield from right_border

    def get_unprocessed_range(self, mat):
        rows, cols = shape(mat)
        valid_rows = range(self.kernel.half_height, rows - self.kernel.half_height)
        valid_cols = range(self.kernel.half_width, cols - self.kernel.half_width)
        return valid_rows, valid_cols

    @staticmethod
    def _handle_border_repeat(mat, row, col):
        rows, cols = shape(mat)
        r_in_bound = max(0, min(row, rows - 1))
        c_in_bound = max(0, min(col, cols - 1))
        return mat[r_in_bound][c_in_bound]

    @staticmethod
    def _handle_border_ignore_zeros(mat, row, col):
        return 0

    def handle(self, mat, row, col):
        return self.func(mat, row, col)


