import os
import sys

import yaml
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
import yolov5.train as train  # noqa: E402
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


@command_component(
    name="train_model",
    display_name="Train model",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def train_model(
    mounted_dataset: Input(type=AssetTypes.URI_FOLDER), yolo_yaml_path: Output(type=AssetTypes.URI_FOLDER)  # type: ignore # noqa: F821
):
    """
    Pipeline step to train the model.

    Parameters
    ----------
    mounted_dataset:
        Dataset to use for training, it should contain the following folder structure:
            - /images/train/
            - /images/val/
            - /images/test/
    yolo_yaml_path:
        Location where to store the yaml file for yolo training.
    """
    data = dict(
        train=f"{mounted_dataset}/images/train/",
        val=f"{mounted_dataset}/images/val/",
        test=f"{mounted_dataset}/images/test/",
        nc=2,
        names=["person", "license_plate"],
    )
    with open(f"{yolo_yaml_path}/yolo_configuration.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    os.system("cp Arial.ttf /root/.config/Ultralytics/Arial.ttf")  # nosec
    train.run(
        data=f"{yolo_yaml_path}/yolo_configuration.yaml",
        weights="../weights/yolov5m.pt",
        cfg="../../../yolov5/models/yolov5s.yaml",
        img=2048,
        batch_size=8,
        epochs=2,
        project="../../../outputs/runs/train",
    )
