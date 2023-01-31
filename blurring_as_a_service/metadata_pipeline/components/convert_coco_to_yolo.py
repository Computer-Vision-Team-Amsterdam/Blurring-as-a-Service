import os
import sys

from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.metadata_pipeline.utils.coco_to_yolo_converter import (  # noqa: E402
    CocoToYoloConverter,
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
    name="convert_coco_to_yolo",
    display_name="Convert coco to yolo",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def convert_coco_to_yolo(
    input_data: Input(type="uri_file"), output_folder: Output(type="uri_folder")  # type: ignore # noqa: F821
):
    CocoToYoloConverter(input_data, output_folder).convert()
