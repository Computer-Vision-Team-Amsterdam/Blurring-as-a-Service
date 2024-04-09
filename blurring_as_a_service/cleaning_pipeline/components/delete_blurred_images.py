import logging
import os
import re
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
log_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)["logging"]
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from cvtoolkit.helpers.file_helpers import delete_file, find_image_paths  # noqa: E402

aml_experiment_settings = settings["aml_experiment_details"]
logger = logging.getLogger("delete_blurred_images")


@command_component(
    name="delete_blurred_images",
    display_name="Deletes from input_structured images that have already been blurred.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def delete_blurred_images(
    _: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    input_structured_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Deletes from input_structured images that have already been blurred.

    Parameters
    ----------
    _:
        Unused folder, it's here just to concatenate the jobs in the pipeline.
        So that this job doesn't start before the previous finishes.
    output_folder:
        Path of the mounted folder containing the blurred images.
    input_structured_folder:
        Path of the mounted folder containing the images to delete.
    """
    for image_path in find_image_paths(output_folder):
        delete_output_images_from_input_structured(image_path, input_structured_folder)


def delete_output_images_from_input_structured(
    output_image_path, input_structured_folder
):
    input_structured_path = replace_azure_uri_folder_path(
        output_image_path, input_structured_folder
    )
    if input_structured_path:
        delete_file(input_structured_path)


def replace_azure_uri_folder_path(path, new_uri_folder_path):
    match = re.search(r"/wd/[^/]+/(.+)", path)
    if match:
        return new_uri_folder_path + "/" + match.group(1)
    else:
        print("There might be an error on the replace_azure_uri_folder_path function.")
