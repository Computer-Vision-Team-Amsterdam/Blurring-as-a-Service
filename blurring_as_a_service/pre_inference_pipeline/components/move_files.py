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
IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)  # image suffixes defined in YOLOv5


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
    # List all files in the mounted folder
    files = os.listdir(input_container)

    if len(files) == 0:
        print("No files in the input zone. Aborting...")
        return  # Skip the rest of the code and exit the function

    target_folder_path = os.path.join(output_container, execution_time)
    # Create the target folder if it doesn't exist
    os.makedirs(target_folder_path, exist_ok=True)

    # Move each file to the target container
    for file_name in files:
        if file_name.lower().endswith(IMG_FORMATS):
            source_file_path = os.path.join(input_container, file_name)
            target_file_path = os.path.join(target_folder_path, file_name)

            # Copy the file to the target directory
            shutil.copy(source_file_path, target_file_path)
            # TODO do we also want to check the max file size?

            # Verify successful file copy
            if not os.path.exists(target_file_path):
                raise FileNotFoundError(
                    f"Failed to move file '{file_name}' to the destination: {target_file_path}"
                )

            # Remove the file from the source folder
            try:
                os.remove(source_file_path)
            except OSError:
                raise OSError(
                    f"Failed to remove file '{file_name}' from the source folder: {source_file_path}"
                )

    print("Files moved and removed successfully.")