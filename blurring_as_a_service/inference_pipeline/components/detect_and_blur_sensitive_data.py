import json
import logging
import os
import secrets
import string
import sys
from datetime import datetime

import torch
import yaml
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

settings = BlurringAsAServiceSettings.set_from_yaml(config_path)

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from cvtoolkit.helpers.file_helpers import delete_file  # noqa: E402
from cvtoolkit.multiprocessing.lock_file import LockFile  # noqa: E402

aml_experiment_settings = settings["aml_experiment_details"]

import yolov5.val as val  # noqa: E402


def generate_unique_string(length):
    # Define the characters to use in the random part of the string
    characters = string.ascii_letters + string.digits

    # Generate a random string of the specified length
    unique_string = "".join(secrets.choice(characters) for _ in range(length))

    return unique_string


def get_current_time():
    current_time = datetime.now()
    current_time_str = current_time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # Format the datetime as a string
    return current_time_str


@command_component(
    name="detect_and_blur_sensitive_data",
    display_name="Detect and blur sensitive data from images",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def detect_and_blur_sensitive_data(
    input_structured_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model: Input(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    batches_files_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    yolo_yaml_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    customer_name: str,
    model_parameters_json: str,
    database_parameters_json: str,
):
    """
    Pipeline step to detect the areas to blur and blur those areas.

    Parameters
    ----------
    input_structured_folder:
        Path of the mounted folder containing the images.
    model:
        Model weights for inference
    batches_files_path:
         Path to folder with multiple text files.
         One text file contains multiple rows.
         Each row is a relative path to {customer_name}_input_structured/inference_queue
    results_path:
        Where to store the results.
    yolo_yaml_path:
        Where to store the yaml file which is used during validation
    customer_name
        The name of the customer, with spaces replaced by underscores
    model_parameters_json
        All parameters used to run YOLOv5 inference in json format
    database_parameters_json
        Database credentials

    """
    # Check if the folder exists
    logger = logging.getLogger("detect_and_blur_sensitive_data")
    if not os.path.exists(batches_files_path):
        raise FileNotFoundError(f"The folder '{batches_files_path}' does not exist.")
    datastore_output_path = settings["inference_pipeline"]["datastore_output_path"]
    if datastore_output_path:
        results_path = os.path.join(results_path, datastore_output_path)
    # Iterate over files in the folder
    for batch_file_txt in os.listdir(batches_files_path):
        if batch_file_txt.endswith(".txt"):
            file_path = os.path.join(batches_files_path, batch_file_txt)

            # Check if the path points to a file (not a directory) and if the file exists
            if os.path.isfile(file_path) and os.path.exists(file_path):
                logger.info(f"Creating inference step: {file_path}")

                files_to_blur_full_path = os.path.join(
                    yolo_yaml_path, batch_file_txt
                )  # use outputs folder as Azure expects outputs there

                try:
                    with LockFile(file_path) as src:
                        with open(files_to_blur_full_path, "w") as dest:
                            for line in src:
                                dest.write(f"{input_structured_folder}/{line}")

                        data = dict(
                            train=f"{files_to_blur_full_path}",
                            val=f"{files_to_blur_full_path}",
                            test=f"{files_to_blur_full_path}",
                            nc=2,
                            names=["person", "license_plate"],
                        )

                        # Remove the extension
                        file_name_without_extension = batch_file_txt.rsplit(".", 1)[0]
                        yaml_name = f"{file_name_without_extension}_pano.yaml"

                        with open(f"{yolo_yaml_path}/{yaml_name}", "w") as outfile:
                            yaml.dump(data, outfile, default_flow_style=False)

                        cuda_device = torch.cuda.current_device()
                        model_parameters = json.loads(model_parameters_json)
                        database_parameters = json.loads(database_parameters_json)
                        val.run(
                            weights=model,
                            data=f"{yolo_yaml_path}/{yaml_name}",
                            project=results_path,
                            device=cuda_device,
                            name="",
                            customer_name=customer_name,
                            start_time=get_current_time(),
                            run_id=generate_unique_string(10),
                            **model_parameters,
                            **database_parameters,
                        )
                        delete_file(files_to_blur_full_path)
                        delete_file(f"{yolo_yaml_path}/{yaml_name}")

                    delete_file(file_path)
                except Exception as e:
                    logger.error(e)
