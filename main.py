from pathlib import Path
import sys
from pipeline_functions import *
from animation_pipeline import create_animation


def main():
    filenames = ["Barcode_custom_1", "Barcode_custom_2"]
    filename = filenames[1]  # just easier to work with

    display_info = {
        "pipeline":
            {
                "show": True,
                "save": True,
            },
        "final_image":
            {
                "show": True,
                "save": True,
            },
        "animation":
            {
                "show": True,  # save needs to be True if this is True
                "save": True,  # eh, shouldn't take more than a song, ~100MB per anim.
            },
    }

    # boring filename/path stuff
    input_filename = "images/" + filename + ".png"

    command_line_arguments = sys.argv[1:]
    if command_line_arguments:
        input_filename = command_line_arguments[0]
        for option in display_info.values():
            option["show"] = False

    images_output_path = Path("output_images")
    if not images_output_path.exists():
        images_output_path.mkdir(parents=True, exist_ok=True)

    final_image_output_filename = images_output_path / Path(filename + "_output.png")
    if len(command_line_arguments) == 2:
        final_image_output_filename = Path(command_line_arguments[1])

    pipeline_filename = images_output_path / Path(filename + "_pipeline.png")

    image_width, image_height, original_image = read_rgb_image(input_filename)
    plot_list = [("Original", original_image)]  # 0

    # step 1: convert to greyscale and normalize
    print("Constructing Greyscale")
    image_grey_scaled = rgb_to_greyscale(original_image)
    plot_list.append(("Greyscale", image_grey_scaled))  # 1

    print("Constructing Normalize")
    image_normalised = normalize(image_grey_scaled,
                                 alpha=(alpha := 0.02), beta=(beta := 0.98))
    plot_list.append((f"Normalized(alpha={alpha}, beta={beta})", image_normalised))  # 2

    # step 2: image gradient method
    print("Constructing Sobel Horizontal Edges")
    image_sobel_filtered_horizontally = sobel_horizontal_edges(image_normalised)
    plot_list.append(("Horizontal Edges", image_sobel_filtered_horizontally))  # 3

    print("Constructing Sobel Vertical Edges")
    image_sobel_filtered_vertically = sobel_vertical_edges(image_normalised)
    plot_list.append(("Vertical Edges", image_sobel_filtered_vertically))  # 4

    print("Constructing Sobel Absolute")
    image_sobel_filtered_absolute = sobel_edges(image_sobel_filtered_horizontally, image_sobel_filtered_vertically)
    plot_list.append(("Sobel Filtered", image_sobel_filtered_absolute))  # 5

    # step 3: Gaussian filter
    print("Constructing Gaussian")
    image_gaussian_filtered = gaussian_filter(image_sobel_filtered_absolute,
                                              size=(size := 9), blur=(blur := 2), iterations=(rounds := 4))
    plot_list.append((f"Gaussian Filtered(size={size}, blur={blur}, rounds={rounds})", image_gaussian_filtered))  # 6

    # step 4: threshold the image
    print("Constructing Threshold")
    image_past_threshold = binary_image_from_threshold(image_gaussian_filtered,
                                                       threshold=(threshold := 0.4))
    plot_list.append((f"Threshold Applied(threshold={threshold})", image_past_threshold))  # 7

    # step 5: erosion and dilation
    print("Constructing Dilation")
    image_dilated = dilate(image_past_threshold,
                           size=(size := 3), iterations=(iterations := 4))  # bridge gaps
    plot_list.append((f"Dilated(size={size}, rounds={iterations})", image_dilated))  # 8

    print("Constructing Erosion")
    image_eroded = erode(image_dilated,
                         size=(size := 2), iterations=(iterations := 10))  # heavy fine detail removal
    plot_list.append((f"Eroded(size={size}, rounds={iterations})", image_eroded))  # 9

    print("Constructing Restoration")
    image_restored = dilate(image_eroded,
                            size=(size := 3), iterations=(iterations := 4))  # partially bring back size of barcode
    plot_list.append((f"Restored(size={size}, rounds={iterations})", image_restored))  # 10

    # step 6: connected component analysis
    print("Constructing Components")
    image_components, components = identify_components(image_restored)
    plot_list.append(("Components Identified", image_components))  # 11

    # step 7: draw a bounding box
    print("Constructing Barcodes")
    barcode_components, image_barcode_components = identify_barcodes(image_width, image_height, components)
    plot_list.append(("Barcodes Filtered", image_barcode_components))  # 12

    print("Constructing Box")
    image_bounded_box, final_image = construct_bounding_box_image(original_image, barcode_components,
                                                                  multi=(multi := True), align=(align := True),
                                                                  bloat=(bloat := 0.2), line_thickness=3)
    plot_list.append((f"Bounding Box Drawn(multi={multi}, align={align}, bloat={bloat})", image_bounded_box))
    plot_list.append((f"Final Result", final_image))  # 13, 14

    # show / save
    print("Compiling Plots")
    construct_pipeline_plot(plot_list)
    if display_info["pipeline"]["save"] is True: plt.savefig(pipeline_filename, bbox_inches="tight", dpi=600)
    if display_info["pipeline"]["show"] is True: plt.show()
    plt.close()
    print("Constructing Animation")
    if display_info["animation"]["save"]: create_animation(plot_list, filename, display_info["animation"]["show"])
    plt.imshow(plot_list[-1][1])
    plt.title(plot_list[-1][0])
    if display_info["final_image"]["save"] is True: plt.savefig(final_image_output_filename, bbox_inches="tight",
                                                                dpi=600)
    if display_info["final_image"]["show"] is True: plt.show()
    plt.close()
    # some bug occurs at the end, who knows


if __name__ == "__main__":
    main()
