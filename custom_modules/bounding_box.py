from .linalgebra import calculate_radius, calculate_mean_point, min_distance_to_line, translate_line, rotate_points
from .helper import deepcopy, create_matrix, pairwise, linspace, shape
from .vector import Vector


class BoundingBox:
    """Arbitrary bounding box class handles operations regarding this box."""

    _eps = 1e-8

    def __init__(self, bbox=None):
        self.bbox = bbox

    def define_bbox_from_region(self, region):
        min_r = max_r = region[0][0]
        min_c = max_c = region[0][1]
        for r, c in region[1:]:
            min_r = min(min_r, r)
            max_r = max(max_r, r)
            min_c = min(min_c, c)
            max_c = max(max_c, c)
        bbox = [(min_r, min_c),  # top left
                (max_r, min_c),  # bottom left
                (max_r, max_c),  # bottom right
                (min_r, max_c)]  # top right
        self.bbox = bbox
        return self

    def get_width_height(self):
        width = abs(self.bbox[2][1] - self.bbox[0][1]) + 1
        height = abs(self.bbox[2][0] - self.bbox[0][0]) + 1
        return width, height

    def get_area(self):
        width, height = self.get_width_height()
        area = width * height
        return area

    def calculate_aspect_ratio(self):
        width, height = self.get_width_height()
        return (width / height) if abs(height) > self._eps else float("inf")

    def copy(self):
        return self.bbox.copy()

    def __getitem__(self, item):
        return self.bbox[item]


class BoundingBoxDrawer:
    """Draws the bounding box around the components on the original image."""

    def __init__(self, multi=False, align=False, bloat=0.0, colour=(255, 0, 0), line_thickness=1, threshold=0.5):
        self.multi = multi
        self.align = align
        self.bloat = bloat
        self.colour = colour
        self.line_thickness = line_thickness
        self.threshold = threshold

    @staticmethod
    def _draw_binary_grid(component, bbox):
        width, height = bbox.get_width_height()
        binary_grid = create_matrix(width, height)

        min_r, min_c = bbox[0]
        for r, c in component:
            binary_grid[r - min_r][c - min_c] = 1

        return binary_grid

    @staticmethod
    def _get_border_pixels(binary_grid, component, bbox):
        border_pixels = set()
        direct = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        min_r, min_c = bbox[0]
        for r, c in component:
            for dr, dc in direct:
                nr, nc = r + dr - min_r, c + dc - min_c
                if nr < 0 or nr >= len(binary_grid) or nc < 0 or nc >= len(binary_grid[0]) or binary_grid[nr][nc] == 0:
                    border_pixels.add((r, c))

        return list(border_pixels)

    @staticmethod
    def _construct_upper_bounding_box(component):
        radius = calculate_radius(component) + 1  # just for safety
        mean_r, mean_c = calculate_mean_point(component)
        min_r = mean_r - radius
        max_r = mean_r + radius
        min_c = mean_c - radius
        max_c = mean_c + radius
        upper_bound_bbox = [(min_r, min_c),  # top left
                            (max_r, min_c),  # bottom left
                            (max_r, max_c),  # bottom right
                            (min_r, max_c)]  # top right
        return BoundingBox(upper_bound_bbox)

    @staticmethod
    def _fit_rectangle(rectangle, border):
        rect = rectangle.copy()

        for i in range(len(rect)):
            side = (rect[i], rect[(i + 1) % len(rect)])
            minimum_distance, closest_point, closest_point_projection = min_distance_to_line(side, border)
            move_vector = (closest_point[0] - closest_point_projection[0],
                           closest_point[1] - closest_point_projection[1])
            new_side = translate_line(side, move_vector)
            rect[i], rect[(i + 1) % len(rect)] = new_side

        return rect

    def _draw_aligned_bounding_boxes(self, image, barcodes):
        """Gets list of 4-tuples representing the corners of an aligned rectangle."""

        barcodes_bboxes = []
        for barcode in barcodes:
            component, bbox = barcode["component"], barcode["bbox"]
            binary_grid = self._draw_binary_grid(component, bbox)
            border_pixels = self._get_border_pixels(binary_grid, component, bbox)
            upper_bound_rectangle = self._construct_upper_bounding_box(component)
            largest_dist = 0
            best_upper_rectangle = upper_bound_rectangle
            angles = linspace(0, 180, 180)
            for angle in angles:  # this is probably not the most efficient method, but eh, computer go brr.
                mean = calculate_mean_point(component)
                rotated_upper_bound_rectangle = rotate_points(upper_bound_rectangle, angle, mean)
                rotated_upper_bound_rectangle = [tuple(round(x) for x in point)
                                                 for point in rotated_upper_bound_rectangle]
                sides = pairwise(rotated_upper_bound_rectangle, end_points=True)
                dist = sum(min_distance_to_line(side, border_pixels)[0] for side in sides)
                if dist > largest_dist:
                    # Short explanation: dist is the total distance the rectangle "gained" when "squished" onto
                    # component. The idea, is that the rectangle who gains the most distance, is probably the one that
                    # aligned with it as closely as possible. Neat aye?
                    largest_dist = dist
                    best_upper_rectangle = rotated_upper_bound_rectangle

            best_bbox = self._fit_rectangle(best_upper_rectangle, border_pixels)
            barcodes_bboxes.append(best_bbox)

        return self._draw_rectangles(image, barcodes_bboxes)

    def _draw_unaligned_bounding_boxes(self, image, barcodes):
        """Gets list of 4-tuples representing the corners of an unaligned rectangle."""

        barcodes_bboxes = [barcode["bbox"] for barcode in barcodes]
        return self._draw_rectangles(image, barcodes_bboxes)

    def _apply_bloat(self, bbox):
        bbox_ = bbox.copy()
        center = calculate_mean_point(bbox_)
        center = Vector(*center)
        for i, corner in enumerate(bbox_):
            direction_vector = Vector(*corner) - center
            bbox_[i] = tuple(map(round, center + direction_vector * (1 + self.bloat)))
        return bbox_

    def _yield_surrounding_points(self, r, c):
        for i in range(1 - self.line_thickness, self.line_thickness):
            for j in range(1 - self.line_thickness, self.line_thickness):
                if abs(i) + abs(j) <= abs(self.line_thickness):
                    yield r + i, c + j

    def _yield_line_points(self, r, c, r_to_travel, c_to_travel, ratio, r_dir, c_dir):
        while r_to_travel > 0 or c_to_travel > 0:
            yield r, c
            yield from self._yield_surrounding_points(r, c)

            current_ratio = (r_to_travel / c_to_travel) if c_to_travel != 0 else float("inf")
            if current_ratio >= ratio:
                r += r_dir
                r_to_travel -= 1
            else:
                c += c_dir
                c_to_travel -= 1

    def _get_line_points(self, point1, point2):
        """Argh, 'tis nae the bonniest, but she'll dae just fine, lassy!"""

        r1, c1 = point1
        r2, c2 = point2
        r_to_travel = abs(r2 - r1)
        c_to_travel = abs(c2 - c1)
        ratio = (r_to_travel / c_to_travel) if c_to_travel != 0 else float("inf")

        r, c = point1
        r_dir = 1 if r2 > r1 else -1
        c_dir = 1 if c2 > c1 else -1

        yield from self._yield_line_points(r, c, r_to_travel, c_to_travel, ratio, r_dir, c_dir)

    def _draw_rectangle(self, image, bbox):
        image = deepcopy(image)
        rows, cols = shape(image)
        sides = list(pairwise(bbox, end_points=True))
        for p1, p2 in sides:
            for r, c in self._get_line_points(p1, p2):
                if 0 <= r < rows and 0 <= c < cols:
                    image[r][c] = self.colour

        return image

    def _draw_rectangles(self, image, bboxes):
        """Given the list of 4-tuple rectangles, it will draw each onto the image and return it."""

        for bbox in bboxes:
            bbox = self._apply_bloat(bbox)
            image = self._draw_rectangle(image, bbox)
            if not self.multi:
                break

        return image

    @staticmethod
    def image_on_image(image_behind, image_top):
        image_behind = deepcopy(image_behind)
        rows, cols = shape(image_behind)
        for r in range(rows):
            for c in range(cols):
                if image_top[r][c] != [0, 0, 0]:
                    image_behind[r][c] = image_top[r][c]
        return image_behind

    def draw_bounding_boxes(self, image, barcodes):
        image = deepcopy(image)
        rows, cols = shape(image)
        blank_image = create_matrix(cols, rows, [0, 0, 0])
        if self.align:
            box_image = self._draw_aligned_bounding_boxes(blank_image, barcodes)
        else:
            box_image = self._draw_unaligned_bounding_boxes(blank_image, barcodes)
        final_image = self.image_on_image(image, box_image)
        return box_image, final_image
