import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Output, command_component

sys.path.append("../../..")
from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

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

azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="split_workload",
    display_name="Distribute the images into multiple batches",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def split_workload(
    data_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    datastore_input_path: str,
    execution_time: str,
    number_of_batches: int,
    exclude_file: str,
    results_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    WorkloadSplitter.create_batches(
        data_folder=data_folder,
        datastore_input_path=datastore_input_path,
        number_of_batches=number_of_batches,
        exclude_file=exclude_file,
        output_folder=results_folder,
        execution_time=execution_time,
    )
