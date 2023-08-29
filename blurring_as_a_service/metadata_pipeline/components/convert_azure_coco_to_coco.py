import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.utils.azure_coco_to_coco_converter import (  # noqa: E402
    AzureCocoToCocoConverter,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


@command_component(
    name="convert_azure_coco_to_coco",
    display_name="Convert Azure coco format to coco",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def convert_azure_coco_to_coco(
    coco_annotations_in: Input(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    coco_annotations_out: Output(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    image_width: int,
    image_height: int,
):
    """
    Pipeline step to convert Azure Coco format to Coco format which is compatible with yolov5 COCO evaluator.

    Extra explanation:
    Annotations were initially rescaled based on the width and height of the images from Data Labelling.
    Since we have multiple image sizes, we want to upscale them to certain widths and heights for a sound comparison.

    Example:
    - original imageA from dataset A is size 2000x4000, thus the absolute, scaled annotations are based on 2000x4000.
    - original imageB from dataset B is size 1000x2000, thus the absolute, scaled annotations are based on 1000x2000.

    We run evaluation on dataset A and we get results.
    We run evaluation on dataset B and we get results.

    However, they are not comparable since the sizes are different.
    If we want a sound comparison, we must use the same scale.

    This is why we introduced the image_width and image_height; they indicate the news sizes of the
    annotations.

    The following is happening for image_width: 8000 and image_height: 4000
    - original imageA from dataset A is size 2000x4000, thus the absolute annotations are rescaled for 4000x8000
    - original imageB from dataset B is size 1000x2000, thus the absolute annotations are rescaled for 4000x8000
    Now we can compare the evaluation for the 2 datasets.


    Parameters
    ----------
    coco_annotations_in: Azure Coco format from Data Labelling
    coco_annotations_out: Coco format which is compatible with yolov5 COCO evaluator
    image_width: new width for absolute coco annotations
    image_height: new height for absolute coco annotations

    Returns
    -------

    """
    AzureCocoToCocoConverter(
        coco_annotations_in,
        coco_annotations_out,
        new_width=image_width,
        new_height=image_height,
    ).convert()
