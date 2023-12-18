import json
import os
import random
import re
import sys
from datetime import datetime

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

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
    customer_cvt_folder:
        Path of the customer data inside the CVT storage account.
    database_parameters_json
        Database credentials
    customer_name
        Customer name
    """
    
    # Find all the images in the input_structured_folder
    image_paths = find_image_paths(input_structured_folder)
    print(f'Input structured folder: {input_structured_folder} \n')
    print(f'Customer Quality Check folder: {customer_quality_check_folder} \n')
    print(f'Customer Retraining folder: {customer_retraining_folder} \n')
    print(f'Number of images found: {len(image_paths)} \n')
    #print(f'Image paths: {image_paths} \n')
    
    # Group the images by date
    grouped_images_by_date = group_files_by_date(image_paths)
    print(f'Number of dates found: {len(grouped_images_by_date)} \n')
    # Print each date
    for key, values in grouped_images_by_date.items():
        print(f"Date: {key} - Number of images: {len(values)}")
    
    # Sample 10 random images for manual quality check
    sample_images_for_quality_check(
        grouped_images_by_date, input_structured_folder, customer_quality_check_folder
    )
    # new_folder = 'test_retraining'
    # new_folder_path = os.path.join(input_structured_folder, new_folder)
    # sample_images_for_quality_check(
    #     grouped_images_by_date, input_structured_folder, new_folder_path
    # )
    
    # Returns a dictionary with the images grouped by date
    images_statistics = collect_images_above_threshold_from_db(
        database_parameters_json, grouped_images_by_date, customer_name
    )
    
    print(f"Images statistics: {images_statistics} \n")
    
    # Count how many images there are for each date (key)
    for key, values in images_statistics.items():
        print(f"Date: {key} - Number of filtered images: {len(values)}")
        
    # Sample .5% of the images for each date
    ratio = 0.5
    sample_images_for_retraining(
        images_statistics, input_structured_folder, customer_retraining_folder, ratio
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
    images_statistics, input_structured_folder, customer_retraining_folder, ratio
):
    # Ratio should be expressed as a decimal for percentage calculation
    percentage_ratio = ratio / 100

    for upload_date, images in images_statistics.items():
        # Calculate the number of images to sample
        num_images_to_sample = int(len(images) * percentage_ratio)

        # If the calculated number is less than 1, we can choose to sample at least 1 image
        num_images_to_sample = max(1, num_images_to_sample)

        # Randomly sample the calculated number of images
        sampled_images = random.sample(images, num_images_to_sample)

        # Copy the sampled images
        for image_name in sampled_images:
            formatted_upload_date = upload_date.strftime("%Y-%m-%d_%H_%M_%S")
            copy_file(
                f"/{formatted_upload_date}/{image_name}", str(input_structured_folder), str(customer_retraining_folder)
            )
            # Optionally, print out the image names being sampled for debugging
            print(f"Sampled for retraining: /{upload_date}/{image_name}")


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


def collect_images_above_threshold_from_db(
    database_parameters_json, grouped_images_by_date, customer_name
):
    
    # 1. Define the threshold for the confidence score
    # Probably best to define it in the config.yml/aml_experiment_details
    conf_score_threshold = 0.0005
    
    # 2. Connect to the database
    database_parameters = json.loads(database_parameters_json)
    db_username = database_parameters["db_username"]
    db_name = database_parameters["db_name"]
    db_hostname = database_parameters["db_hostname"]

    # Validate if database credentials are provided
    if not db_username or not db_name or not db_hostname:
        raise ValueError("Please provide database credentials.")

    # Create a DBConfigSQLAlchemy object
    db_config = DBConfigSQLAlchemy(db_username, db_hostname, db_name)
    # Create the database connection
    db_config.create_connection()

    images_statistics = {}
    
    # TODO: Optimize this code, because now we query the database for each image. Maybe instead
    # we can query first and then filter for all images in image_names
    
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
                ).order_by(DetectionInformation.conf_score.desc())
                
                highest_confidence_result = query.first()
                
                if highest_confidence_result:
                    # Extract the image filename from the highest confidence result
                    highest_confidence_image = highest_confidence_result.image_filename

                    # Add the image to the statistics dictionary
                    if upload_date in images_statistics:
                        if highest_confidence_image not in images_statistics[upload_date]:
                            images_statistics[upload_date].append(highest_confidence_image)
                    else:
                        images_statistics[upload_date] = [highest_confidence_image]

                    # Optional: Print the image name and its highest confidence score for debugging
                    print(f"{upload_date} - {highest_confidence_image}: {highest_confidence_result.conf_score}")
                
                # results = query.all()
                # print(f"Number of results for {image_name} on {upload_date}: {len(results)}")
                

                # if results:
                #     print(f"Sample result for {image_name} on {upload_date}: {results[0].__dict__}")
                #     extracted_data = [result.__dict__ for result in results]
                #     if upload_date in images_statistics:
                #         images_statistics[upload_date].extend(extracted_data)
                #     else:
                #         images_statistics[upload_date] = extracted_data

    return images_statistics