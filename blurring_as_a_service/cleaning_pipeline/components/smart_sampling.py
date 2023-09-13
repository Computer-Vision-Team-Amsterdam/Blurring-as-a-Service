import os
import random
import re
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.utils.generics import (  # noqa: E402
    copy_file,
    find_image_paths,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="smart_sampling",
    display_name="Smart sample images from input_structured",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def smart_sampling(
    input_structured_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    customer_cvt_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to smart sample images from input_structured.
    In order to be able to re-train and evaluate the model.

    Parameters
    ----------
    input_structured_folder:
        Path of the mounted folder containing the images.
    customer_cvt_folder:
        Path of the customer data inside the CVT storage account.
    """
    # TODO: Implement the smart sampling. Ticket: BCV-52:
    # For manual inspection keep 10 images of blurred images.
    # Keep the entire sample from raw images.
    image_paths = find_image_paths(input_structured_folder)
    grouped_images_by_date = group_files_by_date(image_paths)
    quality_check_images = get_10_random_images_per_date(grouped_images_by_date)

    for key, values in quality_check_images.items():
        for value in values:
            copy_file(
                "/" + key + "/" + value, input_structured_folder, customer_cvt_folder
            )


def extract_base_path(path):
    match = re.search(r"^(.+/wd/[^/]+/)", path)
    if match:
        return match.group(1)
    else:
        print("There might be an error on the get_azure_input_path function.")


def group_files_by_date(strings):
    grouped_files = {}

    for string in strings:
        # Use regex to find the date folder pattern
        date_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})", string)
        if date_match:
            date_folder = date_match.group(1)
            # Split the string into key and value parts
            key = date_folder
            value = string[date_match.end(1) + 1 :]
            # Add the entry to the dictionary
            if key in grouped_files:
                grouped_files[key].append(value)
            else:
                grouped_files[key] = [value]

    return grouped_files


def get_10_random_images_per_date(grouped_images_by_date):
    random_result = {}

    for key, values in grouped_images_by_date.items():
        if len(values) >= 10:
            random_values = random.sample(values, 10)
        else:
            random_values = values
        random_result[key] = random_values

    return random_result
