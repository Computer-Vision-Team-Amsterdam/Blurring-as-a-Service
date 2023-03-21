import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.utils.azure_coco_to_yolo_converter import (  # noqa: E402
    AzureCocoToYoloConverter,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


@command_component(
    name="convert_azure_coco_to_yolo",
    display_name="Convert Azure coco format to yolo",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def convert_azure_coco_to_yolo(
    input_data: Input(type=AssetTypes.URI_FILE), output_folder: Output(type=AssetTypes.URI_FOLDER)  # type: ignore # noqa: F821
):
    AzureCocoToYoloConverter(input_data, output_folder).convert()
