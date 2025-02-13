import json
import logging
import sys
from pathlib import Path

from mldesigner import Input, Output, command_component

from blurring_as_a_service import settings

sys.path.append("../../..")

from blurring_as_a_service.performace_evaluation_pipeline.metrics.fnr_calculator import (  # noqa: E402
    FalseNegativeRateCalculator,
)
from blurring_as_a_service.performace_evaluation_pipeline.metrics.tba_calculator import (  # noqa: E402
    collect_and_store_tba_results_per_class_and_size,
)

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="evaluate_with_cvt_metrics",
    display_name="Evaluation with CVT metrics",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def evaluate_with_cvt_metrics(
    mounted_dataset: Input(type="uri_folder"),  # type: ignore # noqa: F821,
    yolo_validation_output: Input(type="uri_folder"),  # type: ignore # noqa: F821,
    coco_annotations: Input(type="uri_file"),  # type: ignore # noqa: F821,
    model_parameters_json: str,
    metrics_metadata_json: str,
    metrics_results: Output(type="uri_folder"),  # type: ignore # noqa: F821
):
    """
    Pipeline step to run TBA and FNR metrics.

    Parameters
    ----------
    mounted_dataset: yolo dataset
    yolo_validation_output: where we store the yolo run results
    coco_annotations: coco annotations from the metadata pipeline output
    model_parameters_json: parameters from the yolo run in json format
    metrics_metadata_json: extra parameters about images to double check if the calculations are sound.
    metrics_results: where to store the files with TBA and FNR metrics.

    Returns
    -------

    """

    logger = logging.getLogger(__name__)
    model_parameters = json.loads(model_parameters_json)
    metrics_metadata = json.loads(metrics_metadata_json)
    save_dir = f"{yolo_validation_output}/{model_parameters['name']}"

    gt_yolo_labels = f"{mounted_dataset}/labels/val"
    dt_yolo_labels = f"{save_dir}/labels"
    dt_yolo_tagged_jsons = f"{save_dir}/labels_tagged"

    # ======== Total Blurred Area metric ========= #
    collect_and_store_tba_results_per_class_and_size(
        gt_yolo_labels,
        dt_yolo_labels,
        markdown_output_path=f"{metrics_results}/tba_results.md",
        image_area=metrics_metadata["image_area"],
    )

    # ======== False Negative Rate metric ========= #

    if Path.exists(Path(dt_yolo_tagged_jsons)):
        metrics_calculator = FalseNegativeRateCalculator(
            dt_yolo_tagged_jsons,
            coco_annotations,
            image_area=metrics_metadata["image_area"],
        )
        metrics_calculator.calculate_and_store_metrics(
            markdown_output_path="fnr_results.md"
        )
    else:
        logger.warning(
            "False Negative Rate metrics can only be run with tagged validation. "
            "labels_tagged folder not found. Make sure you run val.py with the --tagged-data "
            "flag in case you want to compute this metric."
        )
