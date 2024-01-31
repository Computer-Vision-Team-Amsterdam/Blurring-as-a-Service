import logging
import sys
from typing import Dict

import numpy.typing as npt
from tqdm import tqdm

sys.path.append("../../..")

from cvtoolkit.datasets.yolo_labels_dataset import YoloLabelsDataset  # noqa: E402
from cvtoolkit.metrics.total_blurred_area import TotalBlurredArea  # noqa: E402

from blurring_as_a_service.performace_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    ImageSize,
    TargetClass,
    generate_binary_mask,
)

logger = logging.getLogger(__name__)


def get_total_blurred_area_statistics(
    true_labels: Dict[str, npt.NDArray], predicted_labels: Dict[str, npt.NDArray]
):
    """
    Calculates per pixel statistics (tp, tn, fp, fn, precision, recall, f1 score)

    Each key in the dict is an image, each value is a ndarray (n_detections, 5)
    The 6 columns are in the yolo format, i.e. (target_class, x_c, y_c, width, height)

    Parameters
    ----------
    true_labels
    predicted_labels

    Returns
    -------

    """
    total_blurred_area = TotalBlurredArea()

    for image_id in tqdm(true_labels.keys(), total=len(true_labels)):
        tba_true_mask = generate_binary_mask(true_labels[image_id][:, 1:5])
        tba_pred_mask = generate_binary_mask(predicted_labels[image_id][:, 1:5])

        total_blurred_area.update_statistics_based_on_masks(
            true_mask=tba_true_mask, predicted_mask=tba_pred_mask
        )

    results = total_blurred_area.get_statistics()

    return results


def collect_tba_results_per_class_and_size(
    true_path: str, pred_path: str, image_area: int
):
    """

    Computes a dict with statistics (tn, tp, fp, fn, precision, recall, f1) for each target class and size.

    Parameters
    ----------
    true_path
    pred_path
    image_area

    Returns:
    -------

    """
    predicted_dataset = YoloLabelsDataset(folder_path=pred_path, image_area=image_area)
    results = {}

    for target_class in TargetClass:
        for size in ImageSize:
            true_target_class_size = (  # i.e. true_person_small
                YoloLabelsDataset(folder_path=true_path, image_area=image_area)
                .filter_by_class(class_to_keep=target_class.value)
                .filter_by_size(size_to_keep=size.value)
                .get_filtered_labels()
            )
            results[
                f"{target_class.name}_{size.name}"
            ] = get_total_blurred_area_statistics(
                true_target_class_size, predicted_dataset.get_labels()
            )

    return results


def store_tba_results(
    results: Dict[str, Dict[str, float]], markdown_output_path: str = "tba_scores.mda"
):
    """
    Store information from the results dict into a markdown file.
    In this case, the recall from the Total Blurred Area is the only interest number.

    Parameters
    ----------
    results: dictionary with results
    markdown_output_path

    Returns
    -------

    """
    with open(markdown_output_path, "w") as f:
        f.write(
            " Person Small | Person Medium | Person Large |"
            " License Plate Small |  License Plate Medium  | License Plate Large |\n"
        )
        f.write("|----- | ----- |  ----- | ----- | ----- | ----- |\n")
        f.write(
            f'| {results["person_small"]["recall"]} | {results["person_medium"]["recall"]} '
            f'| {results["person_large"]["recall"]}| {results["license_plate_small"]["recall"]} '
            f'| {results["license_plate_medium"]["recall"]} | {results["license_plate_large"]["recall"]}|\n'
        )
        f.write(
            f"Thresholds used for these calculations: Small=`{ImageSize.small.value}`, Medium=`{ImageSize.medium.value}` "
            f"and Large=`{ImageSize.large.value}`."
        )
        f.write(
            f"Thresholds used for these calculations: Small=`{ImageSize.small.value}`, Medium=`{ImageSize.medium.value}` "
            f"and Large=`{ImageSize.large.value}`."
        )


def collect_and_store_tba_results_per_class_and_size(
    ground_truth_path: str,
    predictions_path: str,
    markdown_output_path: str,
    image_area: int,
):
    results: Dict[str, Dict[str, float]] = collect_tba_results_per_class_and_size(
        ground_truth_path, predictions_path, image_area
    )
    store_tba_results(results, markdown_output_path)
