import json
import logging
import os
import sys

import yaml
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.settings.settings_helper import (  # noqa: E402
    setup_azure_logging,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


log_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)["logging"]
setup_azure_logging(log_settings, __name__)

import yolov5.val as val  # noqa: E402


@command_component(
    name="validate_model",
    display_name="Validate model",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def validate_model(
    mounted_dataset: Input(type="uri_folder"),  # type: ignore # noqa: F821
    model: Input(type="uri_file"),  # type: ignore # noqa: F821
    yolo_validation_output: Output(type="uri_folder"),  # type: ignore # noqa: F821
    model_parameters_json: str,
):
    logger = logging.getLogger("validate_model")
    logger.info(f"Hello from {__name__}: {logging.getLogger(__name__).handlers}")
    data = dict(
        train=f"{mounted_dataset}/images/train",
        val=f"{mounted_dataset}/images/val",
        test=f"{mounted_dataset}/images/test",
        nc=2,
        names=["person", "license_plate"],
    )
    with open(f"{yolo_validation_output}/pano.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    os.system("cp Arial.ttf /root/.config/Ultralytics/Arial.ttf")  # nosec
    model_parameters = json.loads(model_parameters_json)

    val.run(
        data=f"{yolo_validation_output}/pano.yaml",
        weights=model,
        project=f"{yolo_validation_output}",  # DO NOT CHANGE
        batch_size=1,  # DO NOT CHANGE
        task="val",  # DO NOT CHANGE
        save_txt=True,  # DO NOT CHANGE
        save_json=True,  # DO NOT CHANGE
        half=True,
        tagged_data=True,
        skip_evaluation=False,
        **model_parameters,
    )
