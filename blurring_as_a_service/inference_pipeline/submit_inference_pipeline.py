import json
import os

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from aml_interface.aml_interface import AMLInterface  # noqa: E402

from blurring_as_a_service.inference_pipeline.components.detect_and_blur_sensitive_data import (  # noqa: E402
    detect_and_blur_sensitive_data,
)


@pipeline()
def inference_pipeline():
    input_structured_path = aml_interface.get_datastore_full_path(
        settings["inference_pipeline"]["datastore_input_structured"]
    )

    input_structured_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_structured_path,
        description="Data to be blurred",
    )

    model_input = Input(
        type=AssetTypes.CUSTOM_MODEL,
        path=f"azureml:{settings['inference_pipeline']['inputs']['model_name']}:{settings['inference_pipeline']['inputs']['model_version']}",
        description="Model weights for evaluation",
    )

    detect_and_blur_sensitive_data_step = detect_and_blur_sensitive_data(
        input_structured_folder=input_structured_input,
        model=model_input,
        customer_name=settings["customer"],
        model_parameters_json=json.dumps(
            settings["inference_pipeline"]["model_parameters"]
        ),
        database_parameters_json=json.dumps(settings["database_parameters"]),
    )

    azureml_outputs_formatted = aml_interface.get_datastore_full_path(
        inference_settings["datastore_output"]
    )
    detect_and_blur_sensitive_data_step.outputs.batches_files_path = Output(
        type="uri_folder",
        mode="rw_mount",
        path=os.path.join(input_structured_path, "inference_queue"),
    )

    detect_and_blur_sensitive_data_step.outputs.results_path = Output(
        type="uri_folder", mode="rw_mount", path=azureml_outputs_formatted
    )
    detect_and_blur_sensitive_data_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=input_structured_path
    )

    return {}


if __name__ == "__main__":
    inference_settings = settings["inference_pipeline"]
    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        inference_pipeline, "inference_pipeline", default_compute
    )
