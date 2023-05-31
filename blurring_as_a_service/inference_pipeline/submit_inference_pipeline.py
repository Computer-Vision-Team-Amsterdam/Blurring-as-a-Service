import json

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.inference_pipeline.components.detect_and_blur_sensitive_data import (
    detect_and_blur_sensitive_data,
)
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def inference_pipeline():
    # Iterate over all customers
    for customer in inference_settings['customers']:
        customer_name = customer['name']

        container_root_formatted = customer['container_root'].format(
            subscription=subscription_id,
            resourcegroup=resource_group,
            workspace=workspace_name,
            datastore_name=f"{customer}_input_structured"
        )

        mounted_root_folder = Input(
            type=AssetTypes.URI_FOLDER,
            path=container_root_formatted,
            description="Data to be blurred",
        )

        files_to_blur_formatted = customer['inputs']['files_to_blur'].format(
            subscription=subscription_id,
            resourcegroup=resource_group,
            workspace=workspace_name,
            datastore_name=f"{customer}_input_structured"
        )

        relative_paths_files_to_blur = Input(
            type=AssetTypes.URI_FILE,
            path=files_to_blur_formatted,
            description="Data to be blurred",
        )

        model_parameters = customer['model_parameters']
        model_parameters_json = json.dumps(model_parameters)  # TODO it seems I can not pass a string to @command_component function

        detect_and_blur_sensitive_data_step = detect_and_blur_sensitive_data(
            mounted_root_folder=mounted_root_folder,
            relative_paths_files_to_blur=relative_paths_files_to_blur,
            customer_name=customer_name,
            model_parameters_json=model_parameters_json
        )

        azureml_outputs_formatted = customer['outputs']['results_path'].format(
            subscription=subscription_id,
            resourcegroup=resource_group,
            workspace=workspace_name,
            datastore_name=f"{customer}_output"
        )

        detect_and_blur_sensitive_data_step.outputs.results_path = Output(
            type="uri_folder", mode="rw_mount", path=azureml_outputs_formatted
        )
        detect_and_blur_sensitive_data_step.outputs.yolo_yaml_path = Output(
            type="uri_folder", mode="rw_mount", path=container_root_formatted
        )

    return {}


def main():
    inference_pipeline_job = inference_pipeline()

    inference_pipeline_job.settings.default_compute = settings[
        "aml_experiment_details"
    ]["compute_name"]

    aml_interface = AMLInterface()
    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=inference_pipeline_job, experiment_name="inference_pipeline"
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
    inference_settings = settings["inference_pipeline"]

    main()
