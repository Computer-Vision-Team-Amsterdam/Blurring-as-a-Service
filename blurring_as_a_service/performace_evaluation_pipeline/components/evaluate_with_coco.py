import json
import os
import sys

from mldesigner import Input, command_component

sys.path.append("../../..")
from blurring_as_a_service.performace_evaluation_pipeline.source.evaluate_with_coco import (  # noqa: E402
    coco_evaluation,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


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
    model_parameters = json.loads(model_parameters_json)
    metrics_metadata = json.loads(metrics_metadata_json)
    coco_predictions = (
        f"{yolo_validation_output}/{model_parameters['name']}/predictions.json"
    )
    coco_evaluation(coco_annotations, coco_predictions, metrics_metadata)
