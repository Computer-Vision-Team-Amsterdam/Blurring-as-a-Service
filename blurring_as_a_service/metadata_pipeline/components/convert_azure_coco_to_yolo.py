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
    is_deterministic=False,
)
def convert_azure_coco_to_yolo(
    coco_annotations_in: Input(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    yolo_annotations: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    tagged_data: bool,
):
    """

    Parameters
    ----------
    coco_annotations_in
    yolo_annotations
    tagged_data

    Returns
    -------

    """
    AzureCocoToYoloConverter(
        coco_annotations_in, yolo_annotations, tagged_data=tagged_data
    ).convert()
