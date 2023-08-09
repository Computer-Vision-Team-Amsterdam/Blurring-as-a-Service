import json
import os
import sys

import torch
import yaml
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
import yolov5.val as val  # noqa: E402
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="detect_and_blur_sensitive_data",
    display_name="Detect and blur sensitive data from images",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def detect_and_blur_sensitive_data(
    mounted_root_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    batch_file_txt: Output(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    yolo_yaml_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    customer_name: str,
    model_parameters_json: str,
):
    """
    Pipeline step to detect the areas to blur.

    Parameters
    ----------
    mounted_root_folder:
        Path of the mounted folder containing the images.
    batch_file_txt:
        Text file containing multiple rows where each row has a relative path,
        taking folder as root and the path to the image.
    results_path:
        Where to store the results.
    yolo_yaml_path:
        Where to store the yaml file which is used during validation
    customer_name
        The name of the customer, with spaces replaced by underscores
    model_parameters_json
        All parameters used to run YOLOv5 inference in json format

    """
    filename = os.path.basename(batch_file_txt)
    files_to_blur_full_path = os.path.join(
        "outputs", filename
    )  # use outputs folder as Azure expects outputs there
    with open(batch_file_txt, "r") as src:
        with open(files_to_blur_full_path, "w") as dest:
            for line in src:
                dest.write(f"{mounted_root_folder}/{line}")
                print(f"{mounted_root_folder}/{line}")

    data = dict(
        train=f"../{files_to_blur_full_path}",
        val=f"../{files_to_blur_full_path}",
        test=f"../{files_to_blur_full_path}",
        nc=2,
        names=["person", "license_plate"],
    )

    # TODO create postgresql string and send to val.py

    with open(f"{yolo_yaml_path}/pano.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    cuda_device = torch.cuda.current_device()
    model_parameters = json.loads(model_parameters_json)
    val.run(
        weights=f"{mounted_root_folder}/best.pt",  # TODO get from Azure ML models
        data=f"{yolo_yaml_path}/pano.yaml",
        project=results_path,
        device=cuda_device,
        name="",
        customer_name=customer_name,  # We want to save this info in a database
        **model_parameters,
    )

    try:
        os.remove(batch_file_txt)
    except OSError as error:
        raise OSError(f"Failed to remove file '{batch_file_txt}': {error}")
