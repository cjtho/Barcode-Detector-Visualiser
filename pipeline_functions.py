import math
from matplotlib import pyplot as plt
from custom_modules.crude_random import RandomNumberGenerator
from custom_modules.helper import create_matrix, deepcopy, shape
from custom_modules.convolution import LinearConvolution
from custom_modules.kernel import Kernel
from custom_modules.morphological import MorphologicalOperation
from custom_modules.barcode_identification import BarcodeIdentifier
from custom_modules.bounding_box import BoundingBoxDrawer


def read_rgb_image(input_filename):
    img = (plt.imread(input_filename) * 255).astype(int)
    image_height, image_width = img.shape[0], img.shape[1]
    image_list = [[(pixel_value[0], pixel_value[1], pixel_value[2]) for pixel_value in row] for row in img]
    return image_width, image_height, image_list


def rgb_to_greyscale(pixel_array_rgb):
    rows, cols = shape(pixel_array_rgb)
    result_array = create_matrix(cols, rows)
    rgb_to_greyscale_ = lambda r_, g_, b_: round(0.299 * r_ + 0.587 * g_ + 0.114 * b_)
    for r, row in enumerate(pixel_array_rgb):
        result_array[r] = [rgb_to_greyscale_(*x) for x in row]
    return result_array


def _get_min_max(pixel_array):
    minimum = min(map(min, pixel_array))
    maximum = max(map(max, pixel_array))
    return minimum, maximum


def _clip(num, lower, upper):
    return max(min(num, upper), lower)


def _min_max_norm(x, minimum, maximum, new_minimum=0, new_maximum=255):
    ratio = (x - minimum) / (maximum - minimum)
    res = round(ratio * (new_maximum - new_minimum) + new_minimum)
    res = _clip(res, new_minimum, new_maximum)
    return _clip(res, new_minimum, new_maximum)


def _get_percentile(pixel_array, percentile):
    flattened = sorted(pixel for row in pixel_array for pixel in row)
    idx = int(len(flattened) * percentile)
    idx = _clip(idx, 0, len(flattened) - 1)
    return flattened[idx]


def normalize(pixel_array, alpha=0.0, beta=1.0):
    rows, cols = shape(pixel_array)
    result_array = create_matrix(cols, rows)
    lower = _get_percentile(pixel_array, alpha)
    upper = _get_percentile(pixel_array, beta)

    if lower == upper:  # edge case
        return result_array

    for r, row in enumerate(pixel_array):
        result_array[r] = [_min_max_norm(val, lower, upper) for val in row]
    return result_array


def sobel_horizontal_edges(pixel_array):
    result_array = deepcopy(pixel_array)
    horizontal_kernel = Kernel([1, 2, 1])
    vertical_kernel = Kernel([-1, 0, 1], vertical=True)
    horizontal_convolution = LinearConvolution(horizontal_kernel, "border_ignore_zeros")
    vertical_convolution = LinearConvolution(vertical_kernel, "border_ignore_zeros", use_abs=True, scaler=1 / 8)
    result_array = horizontal_convolution.apply_convolution(result_array)
    result_array = vertical_convolution.apply_convolution(result_array)
    return result_array


def sobel_vertical_edges(pixel_array):
    result_array = deepcopy(pixel_array)
    horizontal_kernel = Kernel([-1, 0, 1])
    vertical_kernel = Kernel([1, 2, 1], vertical=True)
    horizontal_convolution = LinearConvolution(horizontal_kernel, "border_ignore_zeros")
    vertical_convolution = LinearConvolution(vertical_kernel, "border_ignore_zeros", use_abs=True, scaler=1 / 8)
    result_array = horizontal_convolution.apply_convolution(result_array)
    result_array = vertical_convolution.apply_convolution(result_array)
    return result_array


def sobel_edges(horizontal_array, vertical_array):
    rows, cols = shape(horizontal_array)
    result_array = create_matrix(cols, rows)
    for r in range(rows):
        # technically should use abs(x-y), not max(x,y), but this was interesting so :P
        result_array[r] = [max(horizontal_array[r][c], vertical_array[r][c]) for c in range(cols)]
    return result_array


def gaussian_vector_1d(size, sigma):
    mean = size // 2
    kernel = [math.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) for x in range(size)]
    total = sum(kernel)
    vector = [x / total for x in kernel]
    return vector


def gaussian_filter(pixel_array, size=3, blur=0.85, iterations=1):
    result_array = deepcopy(pixel_array)
    vector = gaussian_vector_1d(size, blur)
    horizontal_kernel = Kernel(vector)
    vertical_kernel = Kernel(vector, vertical=True)
    horizontal_convolution = LinearConvolution(horizontal_kernel, "border_repeat")
    vertical_convolution = LinearConvolution(vertical_kernel, "border_repeat")
    for _ in range(iterations):
        result_array = horizontal_convolution.apply_convolution(result_array)
        result_array = vertical_convolution.apply_convolution(result_array)
    return result_array


def binary_image_from_threshold(pixel_array, threshold):
    rows, cols = shape(pixel_array)
    result_array = create_matrix(cols, rows)
    minimum, maximum = _get_min_max(pixel_array)
    t = int(threshold * (maximum - minimum))
    for r, row in enumerate(pixel_array):
        result_array[r] = [0 if val < t else 1 for val in row]
    return result_array


def dilate(pixel_array, size=3, iterations=1):
    result_array = deepcopy(pixel_array)
    radius = size // 2
    kernel = Kernel([[1 if abs(dr) + abs(dc) <= radius else 0
                      for dc in range(-radius, radius + 1)]
                     for dr in range(-radius, radius + 1)])
    mo = MorphologicalOperation(kernel, "border_repeat", operation=max)
    for _ in range(iterations):
        result_array = mo.operate(result_array)
    return result_array


def erode(pixel_array, size=3, iterations=1):
    result_array = deepcopy(pixel_array)
    radius = size // 2
    kernel = Kernel([[1 if abs(dr) + abs(dc) <= radius else 0
                      for dc in range(-radius, radius + 1)]
                     for dr in range(-radius, radius + 1)])
    mo = MorphologicalOperation(kernel, "border_repeat", operation=min)
    for _ in range(iterations):
        result_array = mo.operate(result_array)
    return result_array


def colour_components(image_width, image_height, components):
    result_array = create_matrix(image_width, image_height, [0, 0, 0])
    rng = RandomNumberGenerator(seed=123)
    for component in components:
        colour = [rng.randint(50, 255) for _ in range(3)]
        for r, c in component:
            result_array[r][c] = colour
    return result_array


def identify_components(pixel_array):
    rows, cols = shape(pixel_array)
    visited = set()

    def dfs(x, y):
        active_pixels = []
        direct = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        stack = [(x, y)]
        visited.add((x, y))
        while stack:
            x, y = stack.pop()
            active_pixels.append((x, y))
            for dx, dy in direct:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue
                if (nx, ny) in visited or pixel_array[nx][ny] == 0:
                    continue
                stack.append((nx, ny))
                visited.add((nx, ny))

        return active_pixels

    components = []  # "id" is key
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited:
                continue
            if pixel_array[r][c] == 0:
                continue
            components.append(dfs(r, c))

    result_array = colour_components(cols, rows, components)
    return result_array, components


def identify_barcodes(image_width, image_height, components):
    barcode_components = BarcodeIdentifier(components).get_barcodes()
    image_barcode_components = colour_components(image_width, image_height, [component_info["component"]
                                                                             for component_info in
                                                                             barcode_components])
    return barcode_components, image_barcode_components


def construct_bounding_box_image(original_image, barcode_components,
                                 multi=False, align=False, bloat=0.0, line_thickness=1):
    bbox_drawer = BoundingBoxDrawer(multi=multi, align=align,
                                    bloat=bloat, line_thickness=line_thickness)
    image_bounded_box, final_image = bbox_drawer.draw_bounding_boxes(original_image, barcode_components)
    return image_bounded_box, final_image


def construct_pipeline_plot(plot_list):
    num_plots = len(plot_list)
    rows = cols = int(num_plots ** 0.5)

    if rows * cols < num_plots: cols += 1
    if rows * cols < num_plots: rows += 1

    fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
    axs = axs.ravel()
    for i, (title, image) in enumerate(plot_list):
        if isinstance(image[0][0], list):  # RGB image
            axs[i].imshow(image)
        else:  # grayscale image
            axs[i].imshow(image, cmap="gray")
        axs[i].set_title(title)
        axs[i].axis("off")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # plt.tight_layout() causes issues for some aspect ratios
