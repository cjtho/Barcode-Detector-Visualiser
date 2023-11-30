class Kernel:
    def __init__(self, kernel, vertical=False):
        if vertical:
            self.kernel = [[x] for x in kernel]
        elif isinstance(kernel[0], (int, float)):
            self.kernel = [kernel]
        else:
            self.kernel = kernel

        self.height = len(self.kernel)
        self.width = len(self.kernel[0])
        self.half_height, self.half_width = self.height // 2, self.width // 2
        self.grid_coordinates = self._get_grid_coordinates()

    def _get_grid_coordinates(self):
        return [(dr, dc)
                for dr in range(-self.half_height, self.half_height + 1)
                for dc in range(-self.half_width, self.half_width + 1)]

    def __getitem__(self, pos):
        r, c = pos
        return self.kernel[self.half_height + r][self.half_width + c]

    def __str__(self):
        kernel_str = "\n".join(str(row) for row in self.kernel)
        return f"{kernel_str}(width={self.width})(height={self.height})"
