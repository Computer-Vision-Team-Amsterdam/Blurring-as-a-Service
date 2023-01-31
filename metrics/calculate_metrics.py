# input:
# val folder with images/ and labels/
# predictions in labels/
import glob

from PIL import Image

from metrics.metrics_utils import generate_mask_binary, parse_labels
from metrics.total_blurred_area import TotalBlurredArea

PATH_TO_IMAGES = "../tests/test-images/val"
PATH_TO_TRUE_LABELS = "../tests/test-labels/val"
PATH_TO_PREDICTED_LABELS = "../tests/test-labels/val"

images = [Image.open(file) for file in glob.glob(f"{PATH_TO_IMAGES}/*.jpg")]
true_labels = [file for file in glob.glob(f"{PATH_TO_TRUE_LABELS}/*.txt")]
pred_labels = [file for file in glob.glob(f"{PATH_TO_PREDICTED_LABELS}/*.txt")]

tba = TotalBlurredArea()

for (
    im,
    true_lbs,
    pred_lbs,
) in zip(images, true_labels, pred_labels):
    true_classes, true_bboxes = parse_labels(true_lbs)
    pred_classes, pred_bboxes = parse_labels(pred_lbs)
    true_mask = generate_mask_binary(true_bboxes, im)
    pred_mask = generate_mask_binary(pred_bboxes, im)

    tba.add_mask(true_mask, pred_mask)

tba.summary()
