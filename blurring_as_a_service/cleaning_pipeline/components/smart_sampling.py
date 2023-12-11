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
    customer_cvt_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
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
    image_paths = find_image_paths(input_structured_folder)
    grouped_images_by_date = group_files_by_date(image_paths)
    # sample_images_for_quality_check(
    #     grouped_images_by_date, input_structured_folder, customer_cvt_folder
    # )
    images_statistics = collect_all_images_statistics_from_db(
        database_parameters_json, grouped_images_by_date, customer_name
    )
    print(images_statistics)


def sample_images_for_quality_check(
    grouped_images_by_date, input_structured_folder, customer_cvt_folder
):
    quality_check_images = get_10_random_images_per_date(grouped_images_by_date)

    for key, values in quality_check_images.items():
        for value in values:
            copy_file(
                "/" + key + "/" + value, input_structured_folder, customer_cvt_folder
            )


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


def collect_all_images_statistics_from_db(
    database_parameters_json, grouped_images_by_date, customer_name
):
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
    with db_config.managed_session() as session:
        for upload_date, image_names in grouped_images_by_date.items():
            upload_date = datetime.strptime(upload_date, "%Y-%m-%d_%H_%M_%S")
            for image_name in image_names:
                query = session.query(DetectionInformation).filter(
                    DetectionInformation.image_customer_name == customer_name,
                    DetectionInformation.image_upload_date == upload_date,
                    DetectionInformation.image_filename == image_name,
                )
                results = query.all()

                if results:
                    extracted_data = [result.__dict__ for result in results]
                    if upload_date in images_statistics:
                        images_statistics[upload_date].extend(extracted_data)
                    else:
                        images_statistics[upload_date] = extracted_data

    return images_statistics
