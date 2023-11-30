from .helper import create_matrix, shape
from .border_handler import BorderHandler


class MorphologicalOperation:

    def __init__(self, kernel, border_mode, operation, use_abs=False, scaler=1):
        self.kernel = kernel
        self.border_handler = BorderHandler(border_mode, kernel)
        self.operation = operation
        self.use_abs = use_abs
        self.scaler = scaler

    def operate(self, mat):
        rows, cols = shape(mat)
        result_mat = create_matrix(cols, rows)

        # process border
        for r, c in self.border_handler.get_border_coordinates(rows, cols):
            vals = []
            for dr, dc in self.kernel.grid_coordinates:
                if self.kernel[dr, dc] == 0:
                    continue
                nr, nc = r + dr, c + dc
                val = self.border_handler.handle(mat, nr, nc)
                vals.append(val)
            result_mat[r][c] = self.operation(vals)

        # process inner image
        unprocessed_rows, unprocessed_cols = self.border_handler.get_unprocessed_range(result_mat)
        for r in unprocessed_rows:
            for c in unprocessed_cols:
                vals = []
                for dr, dc in self.kernel.grid_coordinates:
                    if self.kernel[dr, dc] == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    val = mat[nr][nc]
                    vals.append(val)
                result_mat[r][c] = self.operation(vals)

        return result_mat
