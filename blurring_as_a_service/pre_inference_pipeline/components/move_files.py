import logging
import os
import shutil
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.pre_inference_pipeline.source.image_paths import (  # noqa: E402
    get_image_paths,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.settings.settings_helper import (  # noqa: E402
    setup_azure_logging,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]


log_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)["logging"]
setup_azure_logging(log_settings, __name__)

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)


@command_component(
    name="move_files",
    display_name="Move files",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def move_files(
    execution_time: str,
    input_container: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_container: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to detect the areas to blur.

    Parameters
    ----------
    execution_time:
        Datetime containing when the job was executed. Used to name the folder.
    input_container:
        Path of the mounted root folder containing the images.
    output_container:
        Where to store the results in a Blob Container.

    """
    logger = logging.getLogger("move_files")
    # List all files in the mounted folder and their relative paths
    image_paths = get_image_paths(input_container)

    if len(image_paths) == 0:
        logger.info("No files in the input zone. Aborting...")

    target_folder_path = os.path.join(output_container, execution_time)
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder_path, exist_ok=True)

    # TODO: Refactor this to use the copy_file and delete_file functions.
    #       copy_file(file_name, input_container, target_folder_path)
    #       delete_file(os.path.join(input_container, file_name))
    # Move each file to the target container while preserving the directory structure
    for source_image_path, relative_image_path in image_paths:
        target_file_path = os.path.join(target_folder_path, relative_image_path)

        # Create the directory structure in the target folder if it doesn't exist
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

        # Copy the file to the target directory
        shutil.copy(source_image_path, target_file_path)
        # TODO do we also want to check the max file size?

        # Verify successful file copy
        if not os.path.exists(target_file_path):
            error_message = f"Failed to move file '{relative_image_path}' to the destination: {target_file_path}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        # Remove the file from the source folder
        try:
            os.remove(source_image_path)
        except OSError:
            error_message = f"Failed to remove file '{source_image_path}'."
            logger.error(error_message)
            raise OSError(error_message)

    logger.info("Files moved and removed successfully.")
