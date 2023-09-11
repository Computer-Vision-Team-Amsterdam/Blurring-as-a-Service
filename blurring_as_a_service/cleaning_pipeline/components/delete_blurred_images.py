import os
import re
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)
from yolov5.utils.dataloaders import IMG_FORMATS  # noqa: E402

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


def find_image_paths(root_folder):
    image_paths = []
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in IMG_FORMATS):
                image_path = os.path.join(foldername, filename)
                image_paths.append(image_path)
    return image_paths


def delete_output_images_from_input_structured(
    output_image_path, input_structured_folder
):
    input_structured_path = replace_azure_uri_folder_path(
        output_image_path, input_structured_folder
    )
    if input_structured_path:
        delete_file(input_structured_path)


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"{file_path} has been deleted.")
    except FileNotFoundError as e:
        print(f"{file_path} does not exist.")
        raise FileNotFoundError(f"Failed to remove file '{file_path}': {e}")
    except Exception as e:
        print(f"Failed to remove file '{file_path}': {str(e)}")
        raise (f"Failed to remove file '{file_path}': {e}")


def replace_azure_uri_folder_path(path, new_uri_folder_path):
    match = re.search(r"/wd/[^/]+/(.+)", path)
    if match:
        return new_uri_folder_path + "/" + match.group(1)
    else:
        print("There might be an error on the replace_azure_uri_folder_path function.")
