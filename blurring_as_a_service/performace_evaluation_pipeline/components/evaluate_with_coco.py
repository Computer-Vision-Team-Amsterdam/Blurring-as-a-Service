import json
import sys

from mldesigner import Input, command_component

from blurring_as_a_service import settings

sys.path.append("../../..")

from blurring_as_a_service.performace_evaluation_pipeline.source.evaluate_with_coco import (  # noqa: E402
    coco_evaluation,
)

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="evaluate_with_coco",
    display_name="COCO evaluation",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def evaluate_with_coco(
    coco_annotations: Input(type="uri_file"),  # type: ignore # noqa: F821
    yolo_validation_output: Input(type="uri_folder"),  # type: ignore # noqa: F821
    model_parameters_json: str,
    metrics_metadata_json: str,
):
    """
    Pipeline step to performa COCO evaluation.
    This evaluation helps us obtain mAP, P, R for different sizes: small medium and large.

    Parameters
    ----------
    coco_annotations: Corresponds to metadata_pipeline['outputs']['coco_annotations']
    yolo_validation_output: yolo run results folder. Contains folders such as labels, labels-tagged
    model_parameters_json: parameters from the yolo run in json format
    metrics_metadata_json: extra parameters about images to double check if the calculations are sound.

    Returns
    -------

    """

    model_parameters = json.loads(model_parameters_json)
    metrics_metadata = json.loads(metrics_metadata_json)
    coco_predictions = (
        f"{yolo_validation_output}/{model_parameters['name']}/predictions.json"
    )
    coco_evaluation(coco_annotations, coco_predictions, metrics_metadata)
