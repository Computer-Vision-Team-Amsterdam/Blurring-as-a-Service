import json
import os
import random
import re
import sys
import numpy as np
from datetime import datetime


from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)

from yolov5.baas_utils.database_handler import DBConfigSQLAlchemy  # noqa: E402
from yolov5.baas_utils.database_tables import DetectionInformation  # noqa: E402

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
    print(f'Input structured folder: {input_structured_folder}')
    print(f'Customer Quality Check folder: {customer_quality_check_folder}')
    print(f'Customer Retraining folder: {customer_retraining_folder}')
    print(f'Number of images found: {len(image_paths)}')
    
    # Group the images by date
    grouped_images_by_date = group_files_by_date(image_paths)
    print(f'Number of dates found: {len(grouped_images_by_date)}')
    # Print each date
    for key, values in grouped_images_by_date.items():
        print(f"Date: {key} - Number of images: {len(values)}")
    
    # Sample 10 random images for manual quality check
    sample_images_for_quality_check(
        grouped_images_by_date, input_structured_folder, customer_quality_check_folder
    )
    
    # Returns a dictionary with the images grouped by date and a dictionary to count
    # the number of detections per image
    # TODO: Define the threshold in the config.yml
    conf_score_threshold = 0.0005
    images_statistics, image_counts = collect_images_above_threshold_from_db(
        database_parameters_json, grouped_images_by_date, customer_name, conf_score_threshold
    )
    
    # Calculate and print min and max counts
    if image_counts:
        counts = image_counts.values()
        min_count = min(counts)
        max_count = max(counts)
        print(f"Minimum number of detections for an image: {min_count}")
        print(f"Maximum number of detections for an image: {max_count}")
    else:
        print("No detections found for the given criteria.")
        
    print(f'Image counts: {image_counts}')
    
    print(f'Number of unique images found: {len(image_counts)}')
    
    # Determine the range
    detection_range = max_count - min_count

    # Define a strategy to determine the number of bins based on the range
    if detection_range <= 10:
        bin_size = 3  # For small ranges, fewer bins
    elif 10 < detection_range <= 50:
        bin_size = 5  # Medium ranges get moderate bins
    else:
        bin_size = 10  # Large ranges get more bins

    # Calculate the bin edges
    bins = np.linspace(min_count, max_count, bin_size + 1)

    # Initialize a dictionary to hold bin counts
    bin_counts = {}
    bin_labels = []
    for i in range(len(bins) - 1):
        bin_label = f"{int(bins[i])}-{int(bins[i + 1]) - 1}"
        bin_labels.append(bin_label)
        bin_counts[bin_label] = []

    # Categorize images into bins and count them
    for image, count in image_counts.items():
        bin_index = np.digitize(count, bins, right=True) - 1
        bin_label = bin_labels[bin_index]

        # Add the image to the respective bin
        bin_counts[bin_label].append(image)

    # Count images in each bin
    for bin_label, images in bin_counts.items():
        print(f"Number of images with detections in bin {bin_label}: {len(images)}")
        
    # Sample .5% of the images for each date
    ratio = 0.5
    percentage_ratio = ratio / 100
    
    sampled_images_by_date = sample_images_equally_from_bins(
        image_counts, bin_counts, percentage_ratio
    )
    
    print(f'Sampled images by date: {sampled_images_by_date} \n')
    
    # Sample images for retraining
    sample_images_for_retraining(
        sampled_images_by_date, input_structured_folder, customer_retraining_folder
    )


def sample_images_for_quality_check(
    grouped_images_by_date, input_structured_folder, customer_quality_check_folder
):
    quality_check_images = get_10_random_images_per_date(grouped_images_by_date)
    
    print(f'Quality check images: {quality_check_images} \n')

    for key, values in quality_check_images.items():
        for value in values:
            copy_file(
                "/" + key + "/" + value, str(input_structured_folder), str(customer_quality_check_folder)
            )
            
def sample_images_for_retraining(
    sampled_images_by_date, input_structured_folder, customer_retraining_folder
):
    for upload_date, images in sampled_images_by_date.items():
        # Copy the sampled images
        for image in images:
            formatted_upload_date = upload_date.strftime("%Y-%m-%d_%H_%M_%S")
            image_filename = image[2]  # Assuming image is a tuple (customer_name, upload_date, image_name)
            copy_file(
                f"/{formatted_upload_date}/{image_filename}", str(input_structured_folder), str(customer_retraining_folder)
            )
            # Optionally, print out the image names being sampled for debugging
            print(f"Sampled for retraining: /{formatted_upload_date}/{image_filename}")

def sample_images_equally_from_bins(
    image_counts, bin_counts, percentage_ratio
):
    sampled_images_by_date = {}

    # Extract all unique upload dates from image_counts
    unique_dates = {img[1] for img in image_counts.keys()}  # Extracting the upload_date part of the tuple
    print(f'Unique dates: {unique_dates} \n')

    for upload_date in unique_dates:
        # Filter image_counts for the current date and count unique images
        unique_images_on_date = {k for k in image_counts.keys() if k[1] == upload_date}
        total_images = len(unique_images_on_date)
        print(f'Total images on date {upload_date}: {total_images} \n')
        total_images_to_sample = int(total_images * percentage_ratio)
        
        # Ensure at least one image is sampled if total_images_to_sample > 0
        total_images_to_sample = max(total_images_to_sample, 1) if total_images > 0 else 0
        
        print(f'Total images to sample on date {upload_date}: {total_images_to_sample} \n')

        # Calculate the number of images to sample per bin
        images_per_bin = total_images_to_sample // len(bin_counts)
        print(f'Images per bin: {images_per_bin} \n')
        remainder = total_images_to_sample % len(bin_counts)
        print(f'Remainder: {remainder} \n')

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

def group_files_by_date(strings):
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


def get_10_random_images_per_date(grouped_images_by_date):
    random_result = {}

    for key, values in grouped_images_by_date.items():
        if len(values) >= 10:
            random_values = random.sample(values, 10)
        else:
            random_values = values
        random_result[key] = random_values

    return random_result

def connect_to_database(database_parameters_json):
    """
    Establish a connection to the database.

    Parameters
    ----------
    database_parameters_json : str
        JSON string containing the database credentials.

    Returns
    -------
    db_config : DBConfigSQLAlchemy
        The database configuration object.

    Raises
    ------
    ValueError
        If database credentials are not provided.
    """
    database_parameters = json.loads(database_parameters_json)
    db_username = database_parameters.get("db_username")
    db_name = database_parameters.get("db_name")
    db_hostname = database_parameters.get("db_hostname")

    if not db_username or not db_name or not db_hostname:
        raise ValueError("Please provide database credentials.")

    db_config = DBConfigSQLAlchemy(db_username, db_hostname, db_name)
    return db_config

def collect_images_above_threshold_from_db(
    database_parameters_json, grouped_images_by_date, customer_name, conf_score_threshold
):
    
    images_statistics = {}
    image_counts = {}

    try:
        db_config = connect_to_database(database_parameters_json)
        db_config.create_connection()
    
        with db_config.managed_session() as session:
            for upload_date, image_names in grouped_images_by_date.items():
                print(f'Upload Date: {upload_date} \n')
                upload_date = datetime.strptime(upload_date, "%Y-%m-%d_%H_%M_%S")
                print(f'Formatted Upload Date: {upload_date} \n')
                for image_name in image_names:
                    query = session.query(DetectionInformation).filter(
                        DetectionInformation.image_customer_name == customer_name,
                        DetectionInformation.image_upload_date == upload_date,
                        DetectionInformation.image_filename == image_name,
                        DetectionInformation.conf_score > conf_score_threshold
                    )
                    results = query.all()
                    count = len(results)
                    #print(f"Number of results for {image_name} on {upload_date}: {count}")

                    # Populating image_counts
                    image_key = (customer_name, upload_date, image_name)
                    image_counts[image_key] = count

                    # Populating images_statistics with detailed information
                    if results:
                        extracted_data = [result.__dict__ for result in results]
                        if upload_date in images_statistics:
                            images_statistics[upload_date].extend(extracted_data)
                        else:
                            images_statistics[upload_date] = extracted_data
                            
    except SQLAlchemyError as e:
        print(f"Database operation failed: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return images_statistics, image_counts