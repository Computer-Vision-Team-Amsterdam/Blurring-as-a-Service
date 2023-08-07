import json

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import ManagedIdentityConfiguration

from blurring_as_a_service.inference_pipeline.components.detect_and_blur_sensitive_data import (
    detect_and_blur_sensitive_data,
)
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def inference_pipeline(workspace_name, subscription_id, resource_group, run_id):
    # Call .result() to get the actual values
    workspace_name_actual = workspace_name.result()
    subscription_id_actual = subscription_id.result()
    resource_group_actual = resource_group.result()

    customer_name = inference_settings['customer_name']

    # Format the root path of the Blob Storage Container in Azure using placeholders
    blob_container_path = inference_settings['container_root'].format(
        subscription=subscription_id_actual,
        resourcegroup=resource_group_actual,
        workspace=workspace_name_actual,
        datastore_name=f"{customer_name}_input_structured"
    )

    input_root_folder = Input(
        type=AssetTypes.URI_FOLDER,
        path=blob_container_path,
        description="Data to be blurred",
    )

    # Get the txt file that contains all paths of the files to run inference on
    files_to_blur_path = inference_settings['inputs']['files_to_blur'].format(
        subscription=subscription_id_actual,
        resourcegroup=resource_group_actual,
        workspace=workspace_name_actual,
        datastore_name=f"{customer_name}_input_structured"
    )

    files_to_blur_txt = Input(
        type=AssetTypes.URI_FILE,
        path=files_to_blur_path,
        description="Data to be blurred",
    )

    model_parameters = inference_settings['model_parameters']
    model_parameters_json = json.dumps(model_parameters)  # TODO it seems I can not pass a string to @command_component function

    detect_and_blur_sensitive_data_step = detect_and_blur_sensitive_data(
        mounted_root_folder=input_root_folder,
        relative_paths_files_to_blur=files_to_blur_txt,
        customer_name=customer_name,
        model_parameters_json=model_parameters_json,
        run_id=run_id
    )

    azureml_outputs_formatted = inference_settings['outputs']['results_path'].format(
        subscription=subscription_id_actual,
        resourcegroup=resource_group_actual,
        workspace=workspace_name_actual,
        datastore_name=f"{customer_name}_output"
    )

    detect_and_blur_sensitive_data_step.outputs.results_path = Output(
        type="uri_folder", mode="rw_mount", path=azureml_outputs_formatted
    )
    detect_and_blur_sensitive_data_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=blob_container_path
    )

    return {}


def main():
    aml_interface = AMLInterface()

    # Access the workspace details
    workspace_name = aml_interface.get_workspace_name()
    subscription_id = aml_interface.get_subscription_id()
    resource_group = aml_interface.get_resource_group()

    run_id = aml_interface.get_run_id()

    inference_pipeline_job = inference_pipeline(workspace_name, subscription_id, resource_group, run_id)
    inference_pipeline_job.identity = ManagedIdentityConfiguration()
    inference_pipeline_job.settings.default_compute = settings[
        "aml_experiment_details"
    ]["compute_name"]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=inference_pipeline_job, experiment_name="inference_pipeline"
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    inference_settings = settings["inference_pipeline"]

    main()
