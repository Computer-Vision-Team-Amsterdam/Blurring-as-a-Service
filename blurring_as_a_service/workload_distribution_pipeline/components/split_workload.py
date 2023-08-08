import os
import sys
import json

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.workload_distribution_pipeline.source.workload_splitter import (  # noqa: E402
    WorkloadSplitter,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


@command_component(
    name="split_workload",
    display_name="Distribute the images into multiple batches",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def split_workload(
    data_folder: Input(type=AssetTypes.URI_FOLDER), date_folders_json: str, number_of_batches: int, results_folder: Output(type=AssetTypes.URI_FOLDER)  # type: ignore # noqa: F821
):
    date_folders_list = json.loads(date_folders_json)

    WorkloadSplitter.create_batches(
        data_folder=data_folder,
        number_of_batches=number_of_batches,
        date_folders=date_folders_list,
        output_folder=results_folder,
    )
