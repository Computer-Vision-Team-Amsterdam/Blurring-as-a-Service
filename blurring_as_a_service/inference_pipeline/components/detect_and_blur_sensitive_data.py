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
    relative_paths_files_to_blur: Input(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    model: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    yolo_yaml_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to detect the areas to blur.

    Parameters
    ----------
    mounted_root_folder:
        Path of the mounted folder containing the images.
    relative_paths_files_to_blur:
        Text file containing multiple rows where each row has a relative path,
        taking folder as root and the path to the image.
    model:
        Pre-trained model to be used to blur.
    results_path:
        Where to store the results.
    yolo_yaml_path:
        Where to store the yaml file which is used during validation

    """
    files_to_blur_full_path = "outputs/files_to_blur_full_path.txt"  # use outputs folder as Azure expects outputs there
    with open(relative_paths_files_to_blur, "r") as src:
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

    with open(f"{yolo_yaml_path}/pano.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    cuda_device = torch.cuda.current_device()
    model_parameters = settings["inference_pipeline"]["model_parameters"]
    val.run(
        weights=f"{model}/best.pt",
        data=f"{yolo_yaml_path}/pano.yaml",
        project=results_path,
        batch_size=model_parameters["batch_size"],
        device=cuda_device,
        name="val_detection_results",
        imgsz=model_parameters["img_size"],
        skip_evaluation=True,
        save_blurred_image=True,
    )
