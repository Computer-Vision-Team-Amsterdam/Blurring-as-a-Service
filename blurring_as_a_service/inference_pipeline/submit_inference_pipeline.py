import json
import os

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
    customer_name = settings["customer"]
    model_name = inference_settings["model_name"]
    model_version = inference_settings["model_version"]

    # Format the root path of the Blob Storage Container in Azure using placeholders
    input_structured_path = aml_interface.get_datastore_full_path(
        f"{customer_name}_input_structured"
    )

    input_structured_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_structured_path,
        description="Data to be blurred",
    )

    model_input = Input(
        type=AssetTypes.CUSTOM_MODEL,
        path=f"azureml:{model_name}:{model_version}",
        description="Model weights for evaluation",
    )

    # Get the txt file that contains all paths of the files to run inference on
    batches_files_path = os.path.join(
        input_structured_path,
        "inference_queue",
    )

    model_parameters = inference_settings["model_parameters"]
    model_parameters_json = json.dumps(
        model_parameters
    )  # TODO it seems I can not pass a dict to @command_component function
    database_parameters = settings["database_parameters"]
    database_parameters_json = json.dumps(database_parameters)

    detect_and_blur_sensitive_data_step = detect_and_blur_sensitive_data(
        input_structured_folder=input_structured_input,
        model=model_input,
        customer_name=customer_name,
        model_parameters_json=model_parameters_json,
        database_parameters_json=database_parameters_json,
    )

    azureml_outputs_formatted = aml_interface.get_datastore_full_path(
        f"{customer_name}_output"
    )

    detect_and_blur_sensitive_data_step.outputs.batches_files_path = Output(
        type="uri_folder", mode="rw_mount", path=batches_files_path
    )

    detect_and_blur_sensitive_data_step.outputs.results_path = Output(
        type="uri_folder", mode="rw_mount", path=azureml_outputs_formatted
    )
    detect_and_blur_sensitive_data_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=input_structured_path
    )

    return {}


if __name__ == "__main__":
    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    inference_settings = settings["inference_pipeline"]

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        inference_pipeline, "inference_pipeline", default_compute
    )
