from pathlib import Path

from PIL import Image

from blurring_as_a_service.performace_evaluation_pipeline.metrics.metrics_utils import (
    generate_mask,
    parse_labels,
    visualize_mask,
)


def test_visualize_mask():
    image_path = "../../local_test_data/debug-geo360/images-blurred/00231.jpg"
    stem = Path(image_path).stem
    image = Image.open(image_path)
    gt_labels = "../../local_test_data/debug-geo360/labels/val/00231.txt"
    dt_labels = "../../local_test_data/debug-geo360/labels-pred/00231.txt"

    gt_classes, gt_bounding_boxes = parse_labels(file_path=gt_labels)
    gt_image_out = generate_mask(gt_bounding_boxes, image=image)
    visualize_mask(image=gt_image_out, name=f"{stem}_gt_mask.jpg")

    dt_classes, dt_bounding_boxes = parse_labels(file_path=dt_labels)
    dt_image_out = generate_mask(dt_bounding_boxes, image=image)
    visualize_mask(image=dt_image_out, name=f"{stem}_dt_mask.jpg")
