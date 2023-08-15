import os
from datetime import datetime

from azure.ai.ml import Output
from azure.ai.ml.dsl import pipeline

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
def pre_inference_pipeline(number_of_batches):
    execution_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    for customer in pre_inference_settings["customers"]:
        move_data = move_files(execution_time=execution_time)
        azureml_input_formatted = aml_interface.get_azureml_path(f"{customer}_input")
        azureml_output_formatted = aml_interface.get_azureml_path(
            f"{customer}_input_structured"
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


def main():
    pre_inference_pipeline_job = pre_inference_pipeline(
        number_of_batches=pre_inference_settings["inputs"]["number_of_batches"],
    )
    pre_inference_pipeline_job.settings.default_compute = settings[
        "aml_experiment_details"
    ]["compute_name"]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=pre_inference_pipeline_job,
        experiment_name="pre_inference_pipeline",
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    pre_inference_settings = settings["pre_inference_pipeline"]

    aml_interface = AMLInterface()
    main()
