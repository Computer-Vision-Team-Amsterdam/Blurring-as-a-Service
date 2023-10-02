import os
import shutil
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)
from yolov5.utils.dataloaders import IMG_FORMATS  # noqa: E402


def get_all_files_with_relative_paths(directory):
    # A recursive function to get all files in a directory and its subdirectories
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), directory)
            all_files.append((os.path.join(directory, relative_path), relative_path))
    return all_files


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
    # List all files in the mounted folder and their relative paths
    files = get_all_files_with_relative_paths(input_container)

    if len(files) == 0:
        print("No files in the input zone. Aborting...")

    target_folder_path = os.path.join(output_container, execution_time)
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder_path, exist_ok=True)

    # Move each file to the target container while preserving the directory structure
    for source_file_path, relative_path in files:
        if source_file_path.lower().endswith(IMG_FORMATS):
            target_file_path = os.path.join(target_folder_path, relative_path)

            # Create the directory structure in the target folder if it doesn't exist
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

            # Copy the file to the target directory
            shutil.copy(source_file_path, target_file_path)
            # TODO do we also want to check the max file size?

            # Verify successful file copy
            if not os.path.exists(target_file_path):
                raise FileNotFoundError(
                    f"Failed to move file '{relative_path}' to the destination: {target_file_path}"
                )

            # Remove the file from the source folder
            try:
                os.remove(source_file_path)
            except OSError:
                raise OSError(
                    f"Failed to remove file '{relative_path}'."
                )

    print("Files moved and removed successfully.")
