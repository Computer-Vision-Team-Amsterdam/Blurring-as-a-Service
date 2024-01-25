import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from aml_interface.azure_logging import setup_azure_logging  # noqa: E402

from blurring_as_a_service.pre_inference_pipeline.source.workload_splitter import (  # noqa: E402
    WorkloadSplitter,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
BlurringAsAServiceSettings.set_from_yaml(config_path)
settings = BlurringAsAServiceSettings.get_settings()
setup_azure_logging(settings["logging"], __name__)

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="split_workload",
    display_name="Distribute the images into multiple batches",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def split_workload(
    data_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    execution_time: str,
    number_of_batches: int,
    results_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    WorkloadSplitter.create_batches(
        data_folder=os.path.join(data_folder, execution_time),
        number_of_batches=number_of_batches,
        output_folder=results_folder,
        execution_time=execution_time,
    )
