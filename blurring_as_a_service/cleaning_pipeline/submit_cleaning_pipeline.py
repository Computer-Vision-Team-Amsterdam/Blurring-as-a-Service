from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.cleaning_pipeline.components.delete_blurred_images import (
    delete_blurred_images,
)
from blurring_as_a_service.cleaning_pipeline.components.smart_sampling import (
    smart_sampling,
)
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def cleaning_pipeline():
    customer_name = settings["customer"]

    input_structured_path = aml_interface.get_datastore_full_path(
        f"{customer_name}_input_structured"
    )
    input_structured_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_structured_path,
    )

    customer_in_cvt_path = aml_interface.get_datastore_full_path(
        f"{customer_name}_in_cvt"
    )
    smart_sampling_step = smart_sampling(
        input_structured_folder=input_structured_input,
    )
    smart_sampling_step.outputs.customer_cvt_folder = Output(
        type=AssetTypes.URI_FOLDER,
        mode="rw_mount",
        path=customer_in_cvt_path,
    )

    output_path = aml_interface.get_datastore_full_path(f"{customer_name}_output")
    output_folder_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=output_path,
    )
    delete_blurred_images_step = delete_blurred_images(
        _=smart_sampling_step.outputs.customer_cvt_folder,
        output_folder=output_folder_input,
    )
    delete_blurred_images_step.outputs.input_structured_folder = Output(
        type=AssetTypes.URI_FOLDER,
        mode="rw_mount",
        path=input_structured_path,
    )

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
