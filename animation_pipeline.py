import os
from pathlib import Path
from animation_functions import *
from custom_modules.frame_classes import FrameSaver


def create_animation(plot_list, filename, show):
    frame_directory = Path("animation_intermediate_frames/")
    frame_saver = FrameSaver(filename, frames_dir=frame_directory, show=show, interval=50)

    # images
    original_image = plot_list[0][1]
    image_grey_scaled = plot_list[1][1]
    image_normalised = plot_list[2][1]
    image_sobel_filtered_horizontally = plot_list[3][1]
    image_sobel_filtered_vertically = plot_list[4][1]
    image_sobel_filtered_absolute = plot_list[5][1]
    image_gaussian_filtered = plot_list[6][1]
    image_past_threshold = plot_list[7][1]
    image_dilated = plot_list[8][1]
    image_eroded = plot_list[9][1]
    image_restored = plot_list[10][1]
    image_components = plot_list[11][1]
    image_barcode_components = plot_list[12][1]
    image_bounded_box = plot_list[13][1]
    final_image = plot_list[14][1]

    # animations
    print("Animating RGB to Greyscale")
    animate_rgb_to_greyscale_with_hist(original_image, image_grey_scaled,
                                       frame_saver, total_frames=200)
    print("Animating Greyscale to Normalized")
    animate_greyscale_to_normalized_with_hist(image_grey_scaled, image_normalised,
                                              frame_saver, total_frames=200)
    print("Animating Normalized to Edge Detection")
    animate_normalized_to_edge_detection(image_normalised, image_sobel_filtered_horizontally,
                                         image_sobel_filtered_vertically,
                                         frame_saver, total_frames=150)
    print("Animating Edge Detection to Absolute")
    animate_edge_detection_to_absolute(image_sobel_filtered_horizontally,
                                       image_sobel_filtered_vertically,
                                       image_sobel_filtered_absolute,
                                       frame_saver, total_frames=150)
    print("Animating Absolute to Gaussian")
    animate_absolute_to_gaussian(image_sobel_filtered_absolute, image_gaussian_filtered,
                                 frame_saver, total_frames=50)
    print("Animating Gaussian to Threshold")
    animate_gaussian_to_threshold(image_gaussian_filtered, image_past_threshold,
                                  frame_saver, total_frames=150)
    print("Animating Threshold to Dilation")
    animate_threshold_to_dilation(image_past_threshold, image_dilated,
                                  frame_saver, total_frames=10)
    print("Animating Dilation to Erosion")
    animate_dilation_to_erosion(image_dilated, image_eroded,
                                frame_saver, total_frames=10)
    print("Animating Erosion to Restoration")
    animate_erosion_to_restoration(image_eroded, image_restored,
                                   frame_saver, total_frames=10)
    print("Animating Restoration to Component Identification")
    animate_restoration_to_component_identification(image_restored, image_components,
                                                    frame_saver, total_frames=200)
    print("Animating Barcode Identification")
    animate_component_to_barcode_filter(image_components, image_barcode_components,
                                        frame_saver, total_frames=25)
    print("Animating Box Drawing")
    animate_barcode_filter_to_barcode_filter_with_box(image_barcode_components, image_bounded_box,
                                                      frame_saver, total_frames=50)
    print("Animating Final Image")
    animate_final_image(image_barcode_components, image_bounded_box, final_image,
                        frame_saver, total_frames=50)
    print("Compiling Video")

    frame_saver.create_video_from_frames(video_format="gif")
    if os.path.exists(frame_directory):
        for file in os.listdir(frame_directory):
            file_path = os.path.join(frame_directory, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    for root, dirs, files in os.walk(file_path, topdown=False):
                        for name in files:
                            os.unlink(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(file_path)
            except Exception as e:
                print(f"Error occurred while deleting {file_path}: {e}")

        os.rmdir(frame_directory)
        print("Cleanup complete.")