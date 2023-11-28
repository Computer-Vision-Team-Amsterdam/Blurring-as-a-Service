import os
from datetime import datetime

from azure.ai.ml import Output, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

from blurring_as_a_service.pre_inference_pipeline.components.move_files import (  # noqa: E402
    move_files,
)
from blurring_as_a_service.pre_inference_pipeline.components.split_workload import (
    split_workload,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.utils.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def pre_inference_pipeline():
    execution_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    number_of_batches = settings["pre_inference_pipeline"]["inputs"][
        "number_of_batches"
    ]

    azureml_output_formatted = aml_interface.get_datastore_full_path(
        f"{settings['customer']}_input_structured"
    )

    move_data = move_files(execution_time=execution_time)

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
    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    pre_inference_settings = settings["pre_inference_pipeline"]

    default_compute = pre_inference_settings["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        pre_inference_pipeline, "pre_inference_pipeline", default_compute
    )