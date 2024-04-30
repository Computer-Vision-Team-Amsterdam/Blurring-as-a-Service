import json

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

# from blurring_as_a_service.cleaning_pipeline.components.delete_blurred_images import (
#     delete_blurred_images,
# )
from blurring_as_a_service.cleaning_pipeline.components.smart_sampling import (
    smart_sampling,
)
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from aml_interface.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def cleaning_pipeline():
    customer_name = settings["customer"]

    input_structured_path = aml_interface.get_datastore_full_path(
        f"{customer_name}_input_structured"
    )
    input_structured_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_structured_path,
        description="Path to the folder containing already processed images",
    )

    database_parameters = settings["database_parameters"]
    database_parameters_json = json.dumps(database_parameters)
    smart_sampling_step = smart_sampling(
        input_structured_folder=input_structured_input,
        database_parameters_json=database_parameters_json,
        customer_name=customer_name,
    )

    customer_quality_check_path = aml_interface.get_datastore_full_path(
        f"{customer_name}_quality_check"
    )

    smart_sampling_step.outputs.customer_quality_check_folder = Output(
        type=AssetTypes.URI_FOLDER,
        mode="rw_mount",
        path=customer_quality_check_path,
        description="Path to the folder containing images sampled for quality check",
    )

    customer_retraining_path = aml_interface.get_datastore_full_path(
        f"{customer_name}_retraining"
    )

    smart_sampling_step.outputs.customer_retraining_folder = Output(
        type=AssetTypes.URI_FOLDER,
        mode="rw_mount",
        path=customer_retraining_path,
        description="Path to the folder containing images sampled for retraining the model",
    )

    # output_path = aml_interface.get_datastore_full_path(f"{customer_name}_output")
    # output_folder_input = Input(
    #     type=AssetTypes.URI_FOLDER,
    #     path=output_path,
    #     description="Path to the folder containing the blurred images",
    # )
    # delete_blurred_images_step = delete_blurred_images(
    #     _=smart_sampling_step.outputs.customer_cvt_folder,
    #     output_folder=output_folder_input,
    # )
    # delete_blurred_images_step.outputs.input_structured_folder = Output(
    #     type=AssetTypes.URI_FOLDER,
    #     mode="rw_mount",
    #     path=input_structured_path,
    #     description="Path to the folder containing already processed images",
    # )

    return {}


if __name__ == "__main__":
    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        cleaning_pipeline, "cleaning_pipeline", default_compute
    )
