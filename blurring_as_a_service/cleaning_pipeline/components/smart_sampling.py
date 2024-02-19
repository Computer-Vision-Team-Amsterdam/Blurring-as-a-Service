import logging
import os
import re
import sys
from typing import Dict, List

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]
sampling_parameters = settings["sampling_parameters"]

# Configure logging
# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
log_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)["logging"]
setup_azure_logging(log_settings, __name__)

from blurring_as_a_service.cleaning_pipeline.source.smart_sampler import (  # noqa: E402
    SmartSampler,
)
from blurring_as_a_service.utils.generics import find_image_paths  # noqa: E402

logger = logging.getLogger("smart_sampling")


@command_component(
    name="smart_sampling",
    display_name="Smart sample images from input_structured",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def smart_sampling(
    input_structured_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    customer_quality_check_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    customer_retraining_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    database_parameters_json: str,
    customer_name: str,
):
    """
    Pipeline step to smart sample images from input_structured.
    In order to be able to re-train and evaluate the model.

    Parameters
    ----------
    input_structured_folder:
        Path of the mounted folder containing the images.
    customer_quality_check_folder:
        Path of the customer quality check folder inside the archive storage account.
    customer_retraining_folder:
        Path of the customer retraining folder inside the archive storage account.
    database_parameters_json
        Database credentials
    customer_name
        Customer name
    """

    image_paths = find_image_paths(input_structured_folder)
    logger.info(f"Input structured folder: {input_structured_folder}")
    logger.info(f"Customer Quality Check folder: {customer_quality_check_folder}")
    logger.info(f"Customer Retraining folder: {customer_retraining_folder}")
    logger.info(f"Number of images found: {len(image_paths)}")
    logger.info(f"Sampling parameters: {sampling_parameters}")

    grouped_images_by_date = group_files_by_date(image_paths)
    logger.info(f"Number of dates (folders) found: {len(grouped_images_by_date)}")

    for key, values in grouped_images_by_date.items():
        logger.info(f"Date: {key} - Number of images: {len(values)}")

    smart_sampler = SmartSampler(
        input_structured_folder,
        customer_quality_check_folder,
        customer_retraining_folder,
        database_parameters_json,
        customer_name,
        sampling_parameters,
    )

    smart_sampler.sample_images_for_quality_check(grouped_images_by_date)
    for date in grouped_images_by_date:
        smart_sampler.sample_images_for_retraining(date, grouped_images_by_date[date])


def group_files_by_date(strings: List[str]) -> Dict[str, List[str]]:
    """
    Groups files by date based on their filenames.

    Parameters
    ----------
    strings : List[str]
        A list of strings representing file paths or names.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary where keys are date strings extracted from the filenames,
        and values are lists of filenames belonging to each date.
    """

    grouped_files: Dict[str, List[str]] = {}

    for string in strings:
        # Use regex to find the date folder pattern
        date_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})", string)
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
