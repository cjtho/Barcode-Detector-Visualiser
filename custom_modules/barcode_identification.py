from .bounding_box import BoundingBox


class BarcodeIdentifier:
    """Takes connected components and applies a series of tests to identify the most probably barcodes."""

    _eps = 1e-8

    def __init__(self, components):
        self.components = components

    def _calculate_pixel_density(self, component_area, bbox):
        bbox_area = bbox.get_area()
        return (component_area / bbox_area) if abs(bbox_area) > self._eps else float("inf")

    def _identify_barcodes(self):
        barcodes = []
        largest_barcode_area = 0
        for component in self.components:
            bbox = BoundingBox().define_bbox_from_region(component)
            aspect_ratio = bbox.calculate_aspect_ratio()
            if not (0.56 <= aspect_ratio <= 3):  # include long
                continue
            barcode_area = len(component)
            pixel_density = self._calculate_pixel_density(barcode_area, bbox)
            if not (pixel_density >= 0.4):
                continue
            largest_barcode_area = max(largest_barcode_area, barcode_area)
            barcodes.append({"component": component,
                             "bbox": bbox,
                             "barcode_area": barcode_area})

        return barcodes, largest_barcode_area

    def get_barcodes(self, threshold=0.5):
        barcodes, largest_barcode_area = self._identify_barcodes()
        filtered_bboxes = [barcode for barcode in barcodes
                           if barcode["barcode_area"] >= threshold * largest_barcode_area]
        filtered_bboxes = sorted(filtered_bboxes, key=lambda x: x["barcode_area"], reverse=True)
        return filtered_bboxes
