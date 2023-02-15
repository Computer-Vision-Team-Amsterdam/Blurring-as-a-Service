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
    data_to_blur: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    detect.run(
        weights=f"{model}/best.pt",
        source=data_to_blur,
        project=results_path,
        save_txt=True,
        exist_ok=True,
        name="detection_result",
        imgsz=(2000, 4000),
        # half=True,
        hide_labels=True,
    )
