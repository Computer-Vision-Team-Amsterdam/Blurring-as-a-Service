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
def inference_pipeline(mounted_root_folder, relative_paths_files_to_blur, model):
    customer_list = inference_settings["customers"]
    azureml_input = inference_settings["input_container_root"]
    azureml_output = inference_settings["outputs"]

    # TODO replace {} dingen in yaml vars
    # TODO remove files_to_blur yaml approuch, use something scalable

    for customer in customer_list:
        azureml_input_formatted = azureml_input.format(
            subscription=subscription_id,
            resourcegroup=resource_group,
            workspace=workspace_name,
            datastore_name=f"{customer}_input_structured"
        )

        detect_and_blur_sensitive_data_step = detect_and_blur_sensitive_data(
            mounted_root_folder=mounted_root_folder,
            relative_paths_files_to_blur=relative_paths_files_to_blur,
            model=model,
            customer=customer
        )
        detect_and_blur_sensitive_data_step.outputs.results_path = Output(
            type="uri_folder", mode="rw_mount", path=azureml_output["results_path"]
        )
        detect_and_blur_sensitive_data_step.outputs.yolo_yaml_path = Output(
            type="uri_folder", mode="rw_mount", path=azureml_output["yolo_yaml_path"]
        )

    return {}


def main():
    # TODO move to inference_pipeline
    mounted_root_folder = Input(
        type=AssetTypes.URI_FOLDER,
        path=inference_settings["inputs"]["root_folder"],
        description="Data to be blurred",
    )
    relative_paths_files_to_blur = Input(
        type=AssetTypes.URI_FILE,
        path=inference_settings["inputs"]["files_to_blur"],
        description="Data to be blurred",
    )
    model = Input(
        type=AssetTypes.URI_FOLDER,
        path=inference_settings["inputs"]["model"],
        description="Model to use for the blurring",
    )
    inference_pipeline_job = inference_pipeline(
        mounted_root_folder=mounted_root_folder,
        relative_paths_files_to_blur=relative_paths_files_to_blur,
        model=model,
    )

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
