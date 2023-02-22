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
)
def evaluate_with_coco(
    annotations_json: Input(type="uri_file"), yolo_output_folder: Input(type="uri_folder")  # type: ignore # noqa: F821
):
    coco_evaluation(annotations_json, yolo_output_folder)
