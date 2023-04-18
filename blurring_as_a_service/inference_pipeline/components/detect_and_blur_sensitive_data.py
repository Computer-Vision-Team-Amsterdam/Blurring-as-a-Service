import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
import torch
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
    """
    files_to_blur_full_path = "yolov5/data/pano.yaml"
    print(f"Is het een file? : {os.path.exists(files_to_blur_full_path)}")
    # Read in the file
    with open(files_to_blur_full_path, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('{here}', mounted_root_folder)

    # Write the file out again
    with open(files_to_blur_full_path, 'w') as file:
        file.write(filedata)

    cuda_device = torch.cuda.current_device()
    model_parameters = settings["inference_pipeline"]["model_parameters"]
    val.run(
        weights=f"{model}/best.pt",
        data=files_to_blur_full_path,
        project=results_path,
        # save_txt=model_parameters["save_txt"],
        # exist_ok=model_parameters["exist_ok"],
        batch_size=4,
        device=cuda_device,
        name="detection_result",
        imgsz=4000,
    )
