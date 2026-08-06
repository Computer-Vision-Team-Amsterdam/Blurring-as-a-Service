import os
from datetime import datetime

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from aml_interface.aml_interface import AMLInterface  # noqa: E402

from blurring_as_a_service.pre_inference_pipeline.components.split_workload import (  # noqa: E402
    split_workload,
)


@pipeline()
def pre_inference_pipeline():
    number_of_batches = settings["pre_inference_pipeline"]["inputs"][
        "number_of_batches"
    ]
    exclude_file = settings["pre_inference_pipeline"]["inputs"].get(
        "exclude_list_file", None
    )
    azureml_input_formatted = aml_interface.get_datastore_full_path(
        settings["pre_inference_pipeline"]["datastore_input"]
    )
    azureml_output_formatted = aml_interface.get_datastore_full_path(
        settings["pre_inference_pipeline"]["datastore_output"]
    )
    input_datastore = Input(
        type=AssetTypes.URI_FOLDER,
        path=azureml_input_formatted,
        description="Input datastore",
    )
    split_workload_step = split_workload(
        data_folder=input_datastore,
        datastore_input_path=settings["pre_inference_pipeline"]["datastore_input_path"],
        execution_time=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
        number_of_batches=number_of_batches,
        exclude_file=exclude_file,
    )
    split_workload_step.outputs.results_folder = Output(
        type="uri_folder",
        mode="rw_mount",
        path=os.path.join(azureml_output_formatted, "inference_queue"),
    )
    return {}


aml_interface = AMLInterface()


def main():
    aml_interface.submit_pipeline_experiment(
        pipeline_function=pre_inference_pipeline,
        experiment_name=settings["aml_experiment_details"]["experiment_name"],
        default_compute=settings["aml_experiment_details"]["compute_name"],
        show_log=False,
    )


if __name__ == "__main__":
    main()
