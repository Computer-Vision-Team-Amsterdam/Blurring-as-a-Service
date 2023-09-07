import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="delete_blurred_images",
    display_name="Deletes from input_structured images that have already been blurred.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def delete_blurred_images(
    output_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    input_structured_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Deletes from input_structured images that have already been blurred.

    Parameters
    ----------
    output_folder:
        Path of the mounted folder containing the blurred images.
    input_structured_folder:
        Path of the mounted folder containing the images to delete.
    """

    return
