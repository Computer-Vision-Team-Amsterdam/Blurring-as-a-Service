import os
import sys

from mldesigner import Input, command_component

sys.path.append("../../..")
from blurring_as_a_service.performace_evaluation_pipeline.source.get_data import (  # noqa: E402
    get_and_store_info,
)
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
    name="get_data",
    display_name="Get and store data",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def get_data(input_data: Input(type="uri_folder")):  # type: ignore # noqa: F821
    get_and_store_info(input_data)
