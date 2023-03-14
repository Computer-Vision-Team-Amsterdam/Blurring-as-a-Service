import os
import shutil
import sys
from typing import List

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.inference_pipeline.source.image_blurrer import (  # noqa: E402
    ImageBlurrer,
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
    name="blur_images",
    display_name="Blur images.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def blur_images(
    folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    files_to_blur: Input(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    results_detection: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to blur the images. The areas to blur have already been identified by the model in a previous step.

    Parameters
    ----------
    folder:
        Path of the mounted folder containing the images.
    files_to_blur:
        Text file containing multiple rows where each row has a relative path,
        taking folder as root and the path to the image.
    results_detection:
        Path of the mounted folder containing the result of the detection areas determined by the model.
    results_path:
        Path of the mounted folder where to store the results.
    """
    detection_result_files = [
        os.path.splitext(os.path.basename(path))[0]
        for path in os.listdir(f"{results_detection}/detection_result/labels")
    ]

    with open(files_to_blur, "r") as images_path_to_blur:
        for image_to_blur in images_path_to_blur:
            blur_image(
                image_to_blur,
                detection_result_files,
                folder,
                results_detection,
                results_path,
            )


def blur_image(
    image_to_blur: str,
    detection_result_files: List[str],
    folder: str,
    results_detection: str,
    results_path: str,
):
    """
    Blurs and stores a single image.

    Parameters
    ----------
    image_to_blur
        Relative path to the image to blur.
    detection_result_files
        List of filenames that need to be blurred without the extension.
    folder
        Path of the mounted folder containing the images.
    results_detection
        Path of the mounted folder containing the result of the detection.
    results_path
        Path of the mounted folder where to store the results.
    """
    image_to_blur_filename = os.path.splitext(os.path.basename(image_to_blur))[0]
    if image_to_blur_filename in detection_result_files:
        ImageBlurrer(f"{folder}/{image_to_blur}").blur_and_store(
            f"{results_detection}/detection_result/labels/{image_to_blur_filename}.txt",
            f"{results_path}/detection_result/{image_to_blur}",
        )
    else:
        shutil.copyfile(
            f"{folder}/{image_to_blur}",
            f"{results_path}/detection_result/{image_to_blur}",
        )
