from .border_handler import BorderHandler
from .helper import create_matrix, shape


class LinearConvolution:

    def __init__(self, kernel, border_mode, use_abs=False, scaler=1):
        self.kernel = kernel
        self.border_handler = BorderHandler(border_mode, kernel)
        self.use_abs = use_abs
        self.scaler = scaler

    def apply_convolution(self, mat):
        rows, cols = shape(mat)
        result_mat = create_matrix(cols, rows)

        # process border
        for r, c in self.border_handler.get_border_coordinates(rows, cols):
            val = 0
            for dr, dc in self.kernel.grid_coordinates:
                nr, nc = r + dr, c + dc
                val += self.border_handler.handle(mat, nr, nc) * self.kernel[dr, dc]
            result_mat[r][c] = (abs(val) if self.use_abs else val) * self.scaler
        # process inner image

        unprocessed_rows, unprocessed_cols = self.border_handler.get_unprocessed_range(result_mat)
        for r in unprocessed_rows:
            for c in unprocessed_cols:
                val = 0
                for dr, dc in self.kernel.grid_coordinates:
                    nr, nc = r + dr, c + dc
                    val += mat[nr][nc] * self.kernel[dr, dc]
                result_mat[r][c] = (abs(val) if self.use_abs else val) * self.scaler

        return result_mat
