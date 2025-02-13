from azure.ai.ml.constants import AssetTypes
from cvtoolkit.converters.azure_coco_to_yolo_converter import (  # noqa: E402
    AzureCocoToYoloConverter,
)
from mldesigner import Input, Output, command_component

from blurring_as_a_service import settings

aml_experiment_settings = settings["aml_experiment_details"]


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
    Pipeline step to convert Azure coco annotations to yolo labels.
    Parameters
    ----------
    coco_annotations_in: json file with Azure coco annotations
    yolo_annotations: folder to store the txt files
    tagged_data: whether the yolo labels should contain tagged class id.

    Returns
    -------

    """
    AzureCocoToYoloConverter(
        coco_annotations_in, yolo_annotations, tagged_data=tagged_data
    ).convert()
