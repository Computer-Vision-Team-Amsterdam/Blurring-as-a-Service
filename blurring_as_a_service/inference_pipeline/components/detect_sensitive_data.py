import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
import yolov5.detect as detect  # noqa: E402
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
    name="detect_sensitive_data",
    display_name="Uses a training model to detect sensitive data that needs to be blurred.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def detect_sensitive_data(
    folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    files_to_blur: Input(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    model: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to detect the areas to blur.

    Parameters
    ----------
    data_to_blur:
        Images to be blurred
    model:
        Pre-trained model to be used to blur.
    results_path:
        Where to store the results.
    """
    files_to_blur_full_path = "outputs/files_to_blur_full_path.txt"
    with open(files_to_blur, "r") as src:
        with open(files_to_blur_full_path, "w") as dest:
            for line in src:
                dest.write(f"{folder}/{line}")

    detect.run(
        weights=f"{model}/best.pt",
        source=files_to_blur_full_path,
        project=results_path,
        save_txt=True,
        exist_ok=True,
        name="detection_result",
        imgsz=(2000, 4000),
        # half=True, # Half can be enabled only if run on GPU.
        hide_labels=True,
    )
