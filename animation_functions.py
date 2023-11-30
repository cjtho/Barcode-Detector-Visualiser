import matplotlib.pyplot as plt
from custom_modules.helper import create_matrix, deepcopy, shape
from pipeline_functions import normalize


def compute_frame_interval(total_frames, frames):
    return max(1, total_frames // frames)


def greyscale_to_rgb(img):
    return [[[pixel, pixel, pixel] for pixel in row] for row in img]


def image_to_image_linear_scan(img1, img2):
    img1 = deepcopy(img1)
    rows, cols = shape(img1)
    for r in range(rows):
        for c in range(cols):
            img1[r][c] = img2[r][c]
            yield img1


def generate_histograms(img):
    rows, cols = shape(img)
    hist = [0] * 256
    for r in range(rows):
        for c in range(cols):
            val = int(img[r][c])
            hist[val] += 1
            yield hist


def compute_histogram(img):
    rows, cols = shape(img)
    hist = [0] * 256
    for r in range(rows):
        for c in range(cols):
            val = int(img[r][c])
            hist[val] += 1
    return hist


def hist_to_hist_linear_scan(img1, img2):
    last_hist = compute_histogram(img1)
    rows, cols = shape(img1)
    for r in range(rows):
        for c in range(cols):
            val1 = int(img1[r][c])
            val2 = int(img2[r][c])
            last_hist[val1] -= 1
            last_hist[val2] += 1
            yield last_hist


def image_to_image_fade(img1, img2, iterations):
    img1_ = deepcopy(img1)
    rows, cols = shape(img1_)
    for i in range(iterations + 1):
        alpha = i / iterations
        for r in range(rows):
            for c in range(cols):
                if isinstance(img1[r][c], (int, float)):
                    img1_[r][c] = int((1 - alpha) * img1[r][c] + alpha * img2[r][c])
                else:
                    img1_[r][c] = [int((1 - alpha) * img1[r][c][ch] + alpha * img2[r][c][ch]) for ch in range(3)]
        yield img1_


def image_to_image_vertical_pass(img1, img2):
    img1 = deepcopy(img1)
    rows, cols = shape(img1)
    for c in range(cols):
        for r in range(rows):
            img1[r][c] = img2[r][c]
        yield img1


def image_to_dilated_image(img1, img2):
    img1 = deepcopy(img1)
    rows, cols = shape(img1)
    direct = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    while img1 != img2:
        int_mat = deepcopy(img1)
        for r in range(rows):
            for c in range(cols):
                if img1[r][c] == img2[r][c]:
                    continue
                for dr, dc in direct:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        int_mat[r][c] = max(int_mat[r][c], img1[nr][nc])
        img1 = int_mat
        yield img1


def image_to_eroded_image(img1, img2):
    img1 = deepcopy(img1)
    rows, cols = shape(img1)
    direct = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    while img1 != img2:
        int_mat = deepcopy(img1)
        for r in range(rows):
            for c in range(cols):
                if img1[r][c] == img2[r][c]:
                    continue
                for dr, dc in direct:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        int_mat[r][c] = min(int_mat[r][c], img1[nr][nc])

        img1 = int_mat
        yield img1


def image_to_explored_image(img1, img2):
    img1 = deepcopy(img1)
    rows, cols = shape(img1)
    visited = set()  # just to be sure

    def dfs(row, col):
        active_pixels = []
        direct = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        stack = [(row, col)]
        visited.add((row, col))
        while stack:
            row, col = stack.pop()
            active_pixels.append((row, col))
            for dr, dc in direct:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and img1[nr][nc] != [0, 0, 0]:
                    stack.append((nr, nc))
                    visited.add((nr, nc))

        return active_pixels

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and img1[r][c] == [255, 255, 255]:
                component = dfs(r, c)
                for cr, cc in component:
                    img1[cr][cc] = img2[cr][cc]
                    yield img1


def image_on_top_image_horizontal_pass(img1, img2):
    img1 = deepcopy(img1)
    rows, cols = shape(img1)
    start = False
    for r in range(rows):
        for c in range(cols):
            if img2[r][c] != [0, 0, 0]:
                img1[r][c] = img2[r][c]
                start = True
        if start:
            yield img1


# --- start specific functions ---


def animate_rgb_to_greyscale_with_hist(rgb_image, grey_image, frame_saver, total_frames=100):
    grey_image = normalize(grey_image)

    image_height, image_width = shape(rgb_image)
    frame_interval = compute_frame_interval(image_height * image_width, total_frames)

    fig, axes = plt.subplots(2, 1) if image_width > image_height else plt.subplots(1, 2)

    rgb_image_display = axes[0].imshow(rgb_image)
    rgb_image_display.axes.axis("off")
    axes[0].set_title("Converting to Greyscale")

    histogram_plot = axes[1].bar(range(256), [0] * 256, width=1)
    axes[1].set_xlabel("Greyscale Value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()

    final_histogram = compute_histogram(grey_image)
    axes[1].set_ylim(0, max_hist_height := max(final_histogram))

    image_sequence = image_to_image_linear_scan(rgb_image, grey_image)
    histogram_sequence = generate_histograms(grey_image)
    colormap = plt.cm.get_cmap("inferno")

    for frame_num, (image, hist) in enumerate(zip(image_sequence, histogram_sequence)):
        if frame_num % frame_interval == 0:
            rgb_image_ = [[(pixel, pixel, pixel) if isinstance(pixel, int) else pixel
                           for pixel in row]
                          for row in image]  # scuffed I know
            rgb_image_display.set_array(rgb_image_)

            for bar, height in zip(histogram_plot.patches, hist):
                bar.set_height(height)
                bar.set_color(colormap(height / max_hist_height))

            frame_saver.save_frame(fig)

    plt.close(fig)


def animate_greyscale_to_normalized_with_hist(greyscale_image, normalized_image,
                                              frame_saver, total_frames=100):
    greyscale_image = normalize(greyscale_image)  # the irony
    normalized_image = normalize(normalized_image)

    image_height, image_width = shape(greyscale_image)
    frame_interval = compute_frame_interval(image_height * image_width, total_frames)

    fig, axes = plt.subplots(2, 1) if image_width > image_height else plt.subplots(1, 2)

    greyscale_image_display = axes[0].imshow(greyscale_image, cmap="gray")
    greyscale_image_display.axes.axis("off")
    axes[0].set_title("Normalizing")

    histogram_plot = axes[1].bar(range(256), [0] * 256, width=1)
    axes[1].set_xlabel("Greyscale Value")
    axes[1].set_ylabel("Count")

    plt.tight_layout()

    final_histogram = compute_histogram(greyscale_image)
    axes[1].set_ylim(0, max_hist_height := max(final_histogram))

    image_sequence = image_to_image_linear_scan(greyscale_image, normalized_image)
    histogram_sequence = hist_to_hist_linear_scan(greyscale_image, normalized_image)
    colormap = plt.cm.get_cmap("inferno")

    for frame_num, (image, hist) in enumerate(zip(image_sequence, histogram_sequence)):
        if frame_num % frame_interval == 0:
            greyscale_image_display.set_array(image)
            for bar, height in zip(histogram_plot.patches, hist):
                bar.set_height(height)
                bar.set_color(colormap(height / max_hist_height))
            frame_saver.save_frame(fig)

    plt.close(fig)


def animate_normalized_to_edge_detection(normalized_image, horizontal_edges_image, vertical_edges_image,
                                         frame_saver, total_frames=100):
    normalized_image = normalize(normalized_image)
    horizontal_edges_image = normalize(horizontal_edges_image)
    vertical_edges_image = normalize(vertical_edges_image)

    image_height, image_width = shape(normalized_image)
    frame_interval = compute_frame_interval(image_height * image_width, total_frames)

    fig, axes = plt.subplots(2, 1) if image_width > image_height else plt.subplots(1, 2)

    normalized_image_display = axes[0].imshow(normalized_image, cmap="gray")
    normalized_image_display.axes.axis("off")
    axes[0].set_title("Horizontal Edges")

    horizontal_edges_display = axes[1].imshow(normalized_image, cmap="gray")
    horizontal_edges_display.axes.axis("off")
    axes[1].set_title("Vertical Edges")

    plt.tight_layout()

    horizontal_images_sequence = image_to_image_linear_scan(normalized_image, horizontal_edges_image)
    vertical_images_sequence = image_to_image_linear_scan(normalized_image, vertical_edges_image)

    for frame_num, (horizontal_image, vertical_image) in enumerate(zip(horizontal_images_sequence,
                                                                       vertical_images_sequence)):
        if frame_num % frame_interval == 0:
            normalized_image_display.set_array(horizontal_image)
            horizontal_edges_display.set_array(vertical_image)
            frame_saver.save_frame(fig)

    plt.close(fig)


def animate_edge_detection_to_absolute(horizontal_edges_image, vertical_edges_image, absolute_edges_image,
                                       frame_saver, total_frames=100):
    horizontal_edges_image = normalize(horizontal_edges_image)
    vertical_edges_image = normalize(vertical_edges_image)
    absolute_edges_image = normalize(absolute_edges_image)

    image_height, image_width = shape(horizontal_edges_image)
    frame_interval = compute_frame_interval(image_height * image_width, total_frames)

    blank_image = create_matrix(image_width, image_height)

    fig, axes = plt.subplots(3, 1) if image_width > image_height else plt.subplots(1, 3)

    horizontal_edges_display = axes[0].imshow(horizontal_edges_image, cmap="gray")
    horizontal_edges_display.axes.axis("off")
    axes[0].set_title("Horizontal Edges")

    absolute_edges_display = axes[1].imshow(absolute_edges_image, cmap="gray")
    absolute_edges_display.axes.axis("off")
    axes[1].set_title("Absolute Edges")

    vertical_edges_display = axes[2].imshow(vertical_edges_image, cmap="gray")
    vertical_edges_display.axes.axis("off")
    axes[2].set_title("Vertical Edges")

    plt.tight_layout()

    horizontal_images_sequence = image_to_image_linear_scan(horizontal_edges_image, blank_image)
    absolute_images_sequence = image_to_image_linear_scan(blank_image, absolute_edges_image)
    vertical_images_sequence = image_to_image_linear_scan(vertical_edges_image, blank_image)

    for frame_num, (horizontal_image, absolute_image, vertical_image) in enumerate(zip(horizontal_images_sequence,
                                                                                       absolute_images_sequence,
                                                                                       vertical_images_sequence)):
        if frame_num % frame_interval == 0:
            horizontal_edges_display.set_array(horizontal_image)
            absolute_edges_display.set_array(absolute_image)
            vertical_edges_display.set_array(vertical_image)
            frame_saver.save_frame(fig)

    plt.close(fig)


def animate_absolute_to_gaussian(absolute_edges_image, gaussian_edges_image,
                                 frame_saver, total_frames=100):
    absolute_edges_image = normalize(absolute_edges_image)
    gaussian_edges_image = normalize(gaussian_edges_image)

    fig, ax = plt.subplots()

    absolute_edges_display = ax.imshow(absolute_edges_image, cmap="gray")
    absolute_edges_display.axes.axis("off")
    plt.title("Blurring (Gaussian)")
    plt.tight_layout()

    images_sequence = image_to_image_fade(absolute_edges_image, gaussian_edges_image, total_frames)

    for image in images_sequence:
        absolute_edges_display.set_array(image)
        frame_saver.save_frame(fig)

    plt.close(fig)


def animate_gaussian_to_threshold(gaussian_edges_image, threshold_edges_image,
                                  frame_saver, total_frames=100):
    gaussian_edges_image = normalize(gaussian_edges_image)
    threshold_edges_image = normalize(threshold_edges_image)

    _, image_width = shape(gaussian_edges_image)
    frame_interval = compute_frame_interval(image_width, total_frames)

    fig, axes = plt.subplots()

    gaussian_edges_display = axes.imshow(gaussian_edges_image, cmap="gray")
    gaussian_edges_display.axes.axis("off")
    plt.title("Applying Threshold")

    plt.tight_layout()

    images_sequence = image_to_image_vertical_pass(gaussian_edges_image, threshold_edges_image)

    for frame_num, image in enumerate(images_sequence):
        if frame_num % frame_interval == 0:
            gaussian_edges_display.set_array(image)
            frame_saver.save_frame(fig)

    plt.close(fig)


def animate_threshold_to_dilation(threshold_edges_image, dilation_edges_image,
                                  frame_saver, total_frames=100):
    threshold_edges_image = normalize(threshold_edges_image)
    dilation_edges_image = normalize(dilation_edges_image)

    fig, ax = plt.subplots()

    threshold_edges_display = ax.imshow(threshold_edges_image, cmap="gray")
    threshold_edges_display.axes.axis("off")
    plt.title("Dilating")
    plt.tight_layout()

    previous_image = deepcopy(threshold_edges_image)
    dilation_sequence = image_to_dilated_image(threshold_edges_image, dilation_edges_image)

    for current_image in dilation_sequence:
        fade_sequence = image_to_image_fade(previous_image, current_image, total_frames)
        for interpolation_image in fade_sequence:
            threshold_edges_display.set_array(interpolation_image)
            frame_saver.save_frame(fig)
        previous_image = current_image

    plt.close(fig)


def animate_dilation_to_erosion(dilation_edges_image, erosion_edges_image,
                                frame_saver, total_frames=100):
    dilation_edges_image = normalize(dilation_edges_image)
    erosion_edges_image = normalize(erosion_edges_image)

    fig, ax = plt.subplots()

    dilation_edges_display = ax.imshow(dilation_edges_image, cmap="gray")
    dilation_edges_display.axes.axis("off")
    plt.title("Eroding")
    plt.tight_layout()

    previous_image = deepcopy(dilation_edges_image)
    erosion_sequence = image_to_eroded_image(dilation_edges_image, erosion_edges_image)

    for current_image in erosion_sequence:
        fade_sequence = image_to_image_fade(previous_image, current_image, total_frames)
        for interpolation_image in fade_sequence:
            dilation_edges_display.set_array(interpolation_image)
            frame_saver.save_frame(fig)
        previous_image = current_image

    plt.close(fig)


def animate_erosion_to_restoration(erosion_edges_image, restored_edges_image,
                                   frame_saver, total_frames=100):
    erosion_edges_image = normalize(erosion_edges_image)
    restored_edges_image = normalize(restored_edges_image)

    fig, ax = plt.subplots()

    threshold_edges_display = ax.imshow(erosion_edges_image, cmap="gray")
    threshold_edges_display.axes.axis("off")
    plt.title("Restoring")
    plt.tight_layout()

    previous_image = deepcopy(erosion_edges_image)
    dilation_sequence = image_to_dilated_image(erosion_edges_image, restored_edges_image)

    for current_image in dilation_sequence:
        fade_sequence = image_to_image_fade(previous_image, current_image, total_frames)
        for interpolation_image in fade_sequence:
            threshold_edges_display.set_array(interpolation_image)
            frame_saver.save_frame(fig)
        previous_image = current_image

    plt.close(fig)


def animate_restoration_to_component_identification(restored_edges_image, coloured_components_image,
                                                    frame_saver, total_frames=100):
    restored_edges_image = normalize(restored_edges_image)

    possible_frames = sum(pixel != 0 for row in restored_edges_image for pixel in row)
    frame_interval = compute_frame_interval(possible_frames, total_frames)

    fig, ax = plt.subplots()

    restored_edges_image = greyscale_to_rgb(restored_edges_image)
    restored_edges_display = ax.imshow(restored_edges_image)
    restored_edges_display.axes.axis("off")
    plt.title("Identifying Components (DFS)")
    plt.tight_layout()

    images_sequence = image_to_explored_image(restored_edges_image, coloured_components_image)

    for frame_num, image in enumerate(images_sequence):
        if frame_num % frame_interval == 0:
            restored_edges_display.set_array(image)
            frame_saver.save_frame(fig)

    plt.close(fig)


def animate_component_to_barcode_filter(coloured_components_image, barcode_components_image,
                                        frame_saver, total_frames=100):
    fig, axes = plt.subplots()

    coloured_components_display = axes.imshow(coloured_components_image)
    coloured_components_display.axes.axis("off")
    plt.title("Finding Barcode(s)")

    plt.tight_layout()

    images_sequence = image_to_image_fade(coloured_components_image, barcode_components_image, total_frames)

    for image in images_sequence:
        coloured_components_display.set_array(image)
        frame_saver.save_frame(fig)

    plt.close(fig)


def animate_barcode_filter_to_barcode_filter_with_box(barcodes_image, box_image,
                                                      frame_saver, total_frames=100):
    rows, _ = shape(barcodes_image)
    frame_interval = compute_frame_interval(rows, total_frames)

    fig, axes = plt.subplots()

    barcodes_display = axes.imshow(barcodes_image)
    barcodes_display.axes.axis("off")
    plt.title("Drawing Box(es)")

    plt.tight_layout()

    images_sequence = image_on_top_image_horizontal_pass(barcodes_image, box_image)

    for frame_num, image in enumerate(images_sequence):
        if frame_num % frame_interval == 0:
            barcodes_display.set_array(image)
            frame_saver.save_frame(fig)

    plt.close(fig)


def animate_final_image(barcodes_image, box_image, final_image,
                        frame_saver, total_frames=100):
    last_img = None
    for img in image_on_top_image_horizontal_pass(barcodes_image, box_image):
        last_img = img

    rows, _ = shape(barcodes_image)
    frame_interval = compute_frame_interval(rows, total_frames)

    fig, axes = plt.subplots()

    barcodes_with_box_display = axes.imshow(last_img)
    barcodes_with_box_display.axes.axis("off")
    plt.title("Final Image")

    plt.tight_layout()

    images_sequence = image_on_top_image_horizontal_pass(last_img, final_image)

    for frame_num, image in enumerate(images_sequence):
        if frame_num % frame_interval == 0:
            barcodes_with_box_display.set_array(image)
            frame_saver.save_frame(fig)

    plt.close(fig)
