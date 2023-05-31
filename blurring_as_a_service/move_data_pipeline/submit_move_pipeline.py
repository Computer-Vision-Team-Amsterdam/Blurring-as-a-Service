import json

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
    customer_list = move_data_settings["customers"]
    azureml_input = move_data_settings["input_container_root"]
    azureml_output = move_data_settings["output_container_root"]

    for customer in customer_list:
        move_data = move_files()

        azureml_input_formatted = azureml_input.format(
            subscription=subscription_id,
            resourcegroup=resource_group,
            workspace=workspace_name,
            datastore_name=f"{customer}_input"
        )

        azureml_output_formatted = azureml_output.format(
            subscription=subscription_id,
            resourcegroup=resource_group,
            workspace=workspace_name,
            datastore_name=f"{customer}_input_structured"
        )

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

    aml_interface = AMLInterface()
    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=inference_pipeline_job, experiment_name="move_data_pipeline"
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    # Load the JSON file
    with open('config.json') as f:
        config = json.load(f)

    # Retrieve values from the JSON
    subscription_id = config["subscription_id"]
    resource_group = config["resource_group"]
    workspace_name = config["workspace_name"]

    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    move_data_settings = settings["move_data_pipeline"]

    main()
