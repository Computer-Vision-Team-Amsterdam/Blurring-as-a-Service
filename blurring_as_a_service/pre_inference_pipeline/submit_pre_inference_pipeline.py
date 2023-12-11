import os
from datetime import datetime

from azure.ai.ml import Output
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.settings.settings_helper import setup_azure_logging

# Setting the logger before importing rest of the classes
BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()
setup_azure_logging(settings["logging"], __name__)

from blurring_as_a_service.pre_inference_pipeline.components.move_files import (  # noqa: E402
    move_files,
)
from blurring_as_a_service.pre_inference_pipeline.components.split_workload import (  # noqa: E402
    split_workload,
)
from blurring_as_a_service.utils.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def pre_inference_pipeline():
    execution_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    number_of_batches = settings["pre_inference_pipeline"]["inputs"][
        "number_of_batches"
    ]

    move_data = move_files(execution_time=execution_time)
    azureml_input_formatted = aml_interface.get_datastore_full_path(
        f"{settings['customer']}_input"
    )
    azureml_output_formatted = aml_interface.get_datastore_full_path(
        f"{settings['customer']}_input_structured"
    )

    # NOTE We need to use Output to also delete the files.
    move_data.outputs.input_container = Output(
        type="uri_folder", mode="rw_mount", path=azureml_input_formatted
    )

    move_data.outputs.output_container = Output(
        type="uri_folder", mode="rw_mount", path=azureml_output_formatted
    )

    split_workload_step = split_workload(
        data_folder=move_data.outputs.output_container,
        execution_time=execution_time,
        number_of_batches=number_of_batches,
    )
    split_workload_step.outputs.results_folder = Output(
        type="uri_folder",
        mode="rw_mount",
        path=os.path.join(azureml_output_formatted, "inference_queue"),
    )

    return {}


if __name__ == "__main__":
    pre_inference_settings = settings["pre_inference_pipeline"]
    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        pre_inference_pipeline, "pre_inference_pipeline", default_compute
    )
