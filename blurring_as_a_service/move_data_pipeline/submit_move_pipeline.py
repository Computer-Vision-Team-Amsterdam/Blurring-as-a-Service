from azure.ai.ml import Output
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.move_data_pipeline.components.move_files import (  # noqa: E402
    move_files,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.utils.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def move_files_pipeline():
    for customer in move_data_settings["customers"]:
        move_data = move_files()

        azureml_input_formatted = aml_interface.get_azureml_path(f"{customer}_input")
        azureml_output_formatted = aml_interface.get_azureml_path(f"{customer}_input_structured")

        # NOTE We need to use Output to also delete the files.
        move_data.outputs.input_container = Output(
            type="uri_folder", mode="rw_mount", path=azureml_input_formatted
        )

        move_data.outputs.output_container = Output(
            type="uri_folder", mode="rw_mount", path=azureml_output_formatted
        )

    return {}


def main():
    inference_pipeline_job = move_files_pipeline()
    inference_pipeline_job.settings.default_compute = settings[
        "aml_experiment_details"
    ]["compute_name"]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=inference_pipeline_job, experiment_name="move_data_pipeline"
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    move_data_settings = settings["move_data_pipeline"]

    aml_interface = AMLInterface()

    main()
