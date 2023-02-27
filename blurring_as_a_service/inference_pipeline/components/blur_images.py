import os
import shutil
import sys

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
    data_to_blur: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_detection: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to blur the images. The areas to blur have already been identified by the model in a previous step.

    Parameters
    ----------
    data_to_blur:
        Images to be blurred
    results_detection:
        Areas to blur determined by the model.
    results_path:
        Where to store the results.
    """
    detection_result_files = os.listdir(f"{results_detection}/detection_result/labels")
    detection_result_files = [
        os.path.splitext(os.path.basename(path))[0] for path in detection_result_files
    ]
    for image_to_blur in os.listdir(data_to_blur):
        image_to_blur_filename = os.path.splitext(image_to_blur)[0]
        if image_to_blur_filename in detection_result_files:
            ImageBlurrer(f"{data_to_blur}/{image_to_blur}").blur_and_store(
                f"{results_detection}/detection_result/labels/{image_to_blur_filename}.txt",
                f"{results_path}/detection_result/{os.path.basename(image_to_blur)}",
            )
        else:
            shutil.copyfile(
                f"{data_to_blur}/{image_to_blur}",
                f"{results_path}/detection_result/{os.path.basename(image_to_blur)}",
            )
