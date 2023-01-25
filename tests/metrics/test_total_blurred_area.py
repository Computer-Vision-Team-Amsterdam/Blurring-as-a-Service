from PIL import Image
from metrics.utils import generate_mask_binary, parse_labels
from metrics.total_blurred_area import TotalBlurredArea
from unittest import TestCase


class TestTotalBlurredArea(TestCase):

    image_path = Image.open("../test-images/val/TMX7316010203-000992_pano_0001_000323.jpg")
    true_labels_path = "../test-labels/val/TMX7316010203-000992_pano_0001_000323.txt"
    pred_labels_path = "../test-labels/val/TMX7316010203-000992_pano_0001_000323.txt"

    true_classes, true_bboxes = parse_labels(true_labels_path)
    pred_classes, pred_bboxes = parse_labels(pred_labels_path)
    true_mask = generate_mask_binary(true_bboxes, image_path)
    pred_mask = generate_mask_binary(pred_bboxes, image_path)

    tba = TotalBlurredArea()
    tba.add_mask(true_mask, pred_mask)
    tba.summary()

