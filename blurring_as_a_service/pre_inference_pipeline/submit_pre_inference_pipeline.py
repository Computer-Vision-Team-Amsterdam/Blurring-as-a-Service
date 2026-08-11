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
    input_datastore_fullpath = aml_interface.get_datastore_full_path(
        settings["pre_inference_pipeline"]["inputs"]["datastore"]
    )
    input_datastore = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_datastore_fullpath,
        description="Input datastore",
    )

    split_workload_step = split_workload(
        data_folder=input_datastore,
        input_rel_path=settings["pre_inference_pipeline"]["inputs"]["input_rel_path"],
        execution_time=datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
        number_of_batches=settings["pre_inference_pipeline"]["number_of_batches"],
        exclude_file=settings["pre_inference_pipeline"]["inputs"].get(
            "exclude_list_file", None
        ),
    )

    output_datastore_fullpath = aml_interface.get_datastore_full_path(
        settings["pre_inference_pipeline"]["outputs"]["datastore"]
    )
    inference_queue_folder = os.path.join(
        output_datastore_fullpath,
        settings["pre_inference_pipeline"]["outputs"]["inference_queue_rel_path"],
    )
    split_workload_step.outputs.inference_queue_folder = Output(
        type="uri_folder",
        mode="rw_mount",
        path=inference_queue_folder,
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
