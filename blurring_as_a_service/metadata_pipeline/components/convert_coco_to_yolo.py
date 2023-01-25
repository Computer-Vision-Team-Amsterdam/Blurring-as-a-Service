import sys

from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.metadata_pipeline.utils.coco_to_yolo_converter import (  # noqa: E402
    CocoToYoloConverter,
)


@command_component(
    name="convert_coco_to_yolo",
    display_name="Convert coco to yolo",
    environment="azureml:test-sebastian-env:41",
    code="../../../",
)
def convert_coco_to_yolo(
    input_data: Input(type="uri_file"), output_folder: Output(type="uri_folder")  # type: ignore # noqa: F821
):
    CocoToYoloConverter(input_data, output_folder).convert()
