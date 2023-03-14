import os
import sys

import yaml
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
import yolov5.val as val  # noqa: E402
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
    name="validate_model",
    display_name="Validate model",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def validate_model(
    mounted_dataset: Input(type="uri_folder"),  # type: ignore # noqa: F821
    model: Input(type="uri_folder"),  # type: ignore # noqa: F821
    yolo_yaml_path: Output(type="uri_folder"),  # type: ignore # noqa: F821
    yolo_validation_output: Output(type="uri_folder"),  # type: ignore # noqa: F821
):
    data = dict(
        train=f"{mounted_dataset}/",
        val=f"{mounted_dataset}/",
        test=f"{mounted_dataset}/",
        nc=2,
        names=["person", "license_plate"],
    )
    with open(f"{yolo_yaml_path}/pano.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    os.system("cp Arial.ttf /root/.config/Ultralytics/Arial.ttf")  # nosec
    val.run(
        data=f"{yolo_yaml_path}/pano.yaml",
        weights=f"{model}/last-purple_boot_3l6p24vb.pt",
        imgsz=2048,
        batch_size=8,
        project=f"{yolo_validation_output}",
        task="val",
        save_txt=True,
        save_json=True,
        half=True,
    )
