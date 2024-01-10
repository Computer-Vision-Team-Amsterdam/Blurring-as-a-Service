import json
import os
import random
import re
import sys
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple


from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
from sqlalchemy import func


sys.path.append("../../..")

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.cleaning_pipeline.source.sampling_pipeline import (  # noqa: E402
    SmartSampling,
)
from blurring_as_a_service.settings.settings_helper import (  # noqa: E402
    setup_azure_logging,
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

from blurring_as_a_service.utils.generics import (  # noqa: E402
    copy_file,
    find_image_paths,
)

# Define logger
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
    
    # Find all the images in the input_structured_folder
    image_paths = find_image_paths(input_structured_folder)
    logger.info(f'Input structured folder: {input_structured_folder}')
    logger.info(f'Customer Quality Check folder: {customer_quality_check_folder}')
    logger.info(f'Customer Retraining folder: {customer_retraining_folder}')
    logger.info(f'Number of images found: {len(image_paths)}')
    logger.info(f'Sampling parameters: {sampling_parameters}')
    
    # Group the images by date
    grouped_images_by_date = group_files_by_date(image_paths)
    logger.info(f'Number of dates (folder) found: {len(grouped_images_by_date)}')
    
    # Log each date
    for key, values in grouped_images_by_date.items():
        logger.info(f"Date: {key} - Number of images: {len(values)}")
    
    # Define the SmartSampling object
    smartSampling = SmartSampling(input_structured_folder, customer_quality_check_folder, customer_retraining_folder, database_parameters_json,
                                            customer_name, sampling_parameters)
    
    # Sample a number of random images for manual quality check
    smartSampling.sample_images_for_quality_check(grouped_images_by_date, input_structured_folder, customer_quality_check_folder)

    # Collect images above the confidence score threshold from the database
    _, image_counts = smartSampling.collect_images_above_threshold_from_db(database_parameters_json, grouped_images_by_date, customer_name)
    
    # Group images into bins
    bin_counts, _ = categorize_images_into_bins(image_counts)

    # Count images in each bin
    for bin_label, images in bin_counts.items():
        logger.info(f"Number of images with detections in bin {bin_label}: {len(images)}")
        
    # Sample a ratio of the images for each date
    # The ratio is set in config.yml as sampling_ratio
    ratio = sampling_parameters["sampling_ratio"]
    percentage_ratio = ratio / 100
    
    sampled_images_by_date = sample_images_equally_from_bins(
        image_counts, bin_counts, percentage_ratio
    )
    
    logger.info(f'Sampled images by date: {sampled_images_by_date} \n')
    
    # Sample images for retraining
    smartSampling.sample_images_for_retraining(sampled_images_by_date)

def categorize_images_into_bins(image_counts: Dict[str, int]) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Categorizes images into bins based on their detection counts.

    Parameters
    ----------
    image_counts : Dict[str, int]
        A dictionary mapping image identifiers to their respective detection counts.

    Returns
    -------
    Tuple[Dict[str, List[str]], List[str]]
        A tuple containing two elements:
        - A dictionary where keys are bin labels and values are lists of image identifiers in that bin.
        - A list of bin labels.

    If no detections are found, an empty dictionary and list are returned.
    """
    
    if not image_counts:
        logger.info("No detections found for the given criteria.")
        return {}, []

    # Calculate min and max counts
    counts = image_counts.values()
    min_count, max_count = min(counts), max(counts)
    logger.info(f"Minimum number of detections for an image: {min_count}")
    logger.info(f"Maximum number of detections for an image: {max_count}")

    # Determine the range and define bin size strategy
    detection_range = max_count - min_count
    bin_size = determine_bin_size(detection_range)

    # Calculate the bin edges
    bins = np.linspace(min_count, max_count, bin_size + 1)

    # Initialize a dictionary to hold bin counts
    bin_counts, bin_labels = initialize_bin_counts(bins)

    # Categorize images into bins
    categorize_into_bins(image_counts, bins, bin_labels, bin_counts)

    return bin_counts, bin_labels

def determine_bin_size(detection_range: int) -> int:
    """
    Determines the bin size for categorization based on the detection range.

    Parameters
    ----------
    detection_range : int
        The range of detection counts across all images.

    Returns
    -------
    int
        The number of bins to be used for categorization.
    """
    
    if detection_range <= 10:
        return 3
    elif 10 < detection_range <= 50:
        return 5
    else:
        return 10

def initialize_bin_counts(bins: np.ndarray) -> Tuple[Dict[str, List], List[str]]:
    """
    Initializes a dictionary (bin_counts) and a list (bin_labels) that will be used 
    for categorizing images into bins based on their detection counts. 

    Parameters
    ----------
    bins : np.ndarray
        The array of bin edges.

    Returns
    -------
    Tuple[Dict[str, List], List[str]]
        A tuple containing two elements:
        - A dictionary where keys are bin labels and values are empty lists for each bin.
        - A list of bin labels.
    """
    
    bin_counts = {}
    
    # Iterate through the given bins array and create a list of bin labels. 
    # Each label represents a range, formatted as "start-end", 
    # where "start" is the beginning of a bin and "end" is one less than the start of the next bin.
    bin_labels = [f"{int(bins[i])}-{int(bins[i + 1]) - 1}" for i in range(len(bins) - 1)]
    
    # Create a dictionary (bin_counts) with keys being the bin labels and values being empty lists. 
    # Each list will contain image identifiers that fall within the corresponding bin's range. 
    for label in bin_labels:
        bin_counts[label] = []
        
    return bin_counts, bin_labels

def categorize_into_bins(image_counts: Dict[str, int], bins: np.ndarray, bin_labels: List[str], bin_counts: Dict[str, List[str]]) -> None:
    """
    Categorizes each image into a bin based on its detection count.

    Parameters
    ----------
    image_counts : Dict[str, int]
        A dictionary mapping image identifiers to their detection counts.
    bins : np.ndarray
        The array of bin edges.
    bin_labels : List[str]
        A list of bin labels.
    bin_counts : Dict[str, List[str]]
        A dictionary to hold the categorized images, with keys as bin labels and values as lists of images.

    Returns
    -------
    None
        This function modifies the bin_counts dictionary in place.
    """
    
    for image, count in image_counts.items():
        bin_index = np.digitize(count, bins, right=True) - 1
        bin_label = bin_labels[bin_index]
        bin_counts[bin_label].append(image)

def sample_images_equally_from_bins(
    image_counts: Dict[Tuple[str, datetime, str], int], 
    bin_counts: Dict[str, List[Tuple[str, datetime, str]]], 
    percentage_ratio: float
) -> Dict[datetime, List[Tuple[str, datetime, str]]]:
    """
    Samples a percentage of images equally from each bin for each date.

    Parameters
    ----------
    image_counts : Dict[Tuple[str, datetime, str], int]
        A dictionary mapping image tuples to their detection counts. Each tuple contains customer name, 
        upload date, and image name.
    bin_counts : Dict[str, List[Tuple[str, datetime, str]]]
        A dictionary where keys are bin labels and values are lists of image tuples in that bin.
    percentage_ratio : float
        The ratio of total images to sample from each date.

    Returns
    -------
    Dict[datetime, List[Tuple[str, datetime, str]]]
        A dictionary mapping dates to lists of sampled image tuples.
    """
    
    sampled_images_by_date = {}

    # Extract all unique upload dates from image_counts
    unique_dates = {img[1] for img in image_counts.keys()}  # Extracting the upload_date part of the tuple
    logger.info(f'Unique dates: {unique_dates}')

    for upload_date in unique_dates:
        # Filter image_counts for the current date and count unique images
        unique_images_on_date = {k for k in image_counts.keys() if k[1] == upload_date}
        total_images = len(unique_images_on_date)
        logger.info(f'Total images on date {upload_date}: {total_images}')
        total_images_to_sample = int(total_images * percentage_ratio)
        
        # Ensure at least one image is sampled if total_images_to_sample > 0
        total_images_to_sample = max(total_images_to_sample, 1) if total_images > 0 else 0
        
        logger.info(f'Total images to sample on date {upload_date}: {total_images_to_sample}')

        # Calculate the number of images to sample per bin
        images_per_bin = total_images_to_sample // len(bin_counts)
        logger.info(f'Images per bin: {images_per_bin}')
        remainder = total_images_to_sample % len(bin_counts)
        logger.info(f'Remainder: {remainder}')

        sampled_images = []

        for bin_label, images_in_bin in bin_counts.items():
            # Extract the unique images for the current bin and date
            unique_images_in_bin = [img for img in images_in_bin if img[1] == upload_date]
            
            # Adjust the number of images to sample from this bin
            if images_per_bin > 0 or (remainder > 0 and unique_images_in_bin):
                num_to_sample = min(images_per_bin + (1 if remainder > 0 else 0), len(unique_images_in_bin))
                remainder -= 1 if remainder > 0 else 0
            else:
                # If images_per_bin is 0 and no remainder, skip this bin
                continue 

            # Sample images
            sampled_from_bin = random.sample(unique_images_in_bin, num_to_sample) if unique_images_in_bin else []
            sampled_images.extend(sampled_from_bin)

        sampled_images_by_date[upload_date] = sampled_images

    return sampled_images_by_date

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
    
    grouped_files = {}

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
