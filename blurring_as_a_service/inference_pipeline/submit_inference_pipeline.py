from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.inference_pipeline.components.blur_images import blur_images
from blurring_as_a_service.inference_pipeline.components.detect_sensitive_data import (
    detect_sensitive_data,
)
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def inference_pipeline(data_to_blur, model):
    outputs = BlurringAsAServiceSettings.get_settings()["inference_pipeline"]["outputs"]

    detect_sensitive_data_step = detect_sensitive_data(
        data_to_blur=data_to_blur, model=model
    )
    detect_sensitive_data_step.outputs.results_path = Output(
        type="uri_folder", mode="rw_mount", path=outputs["results_path"]
    )

    blur_images_step = blur_images(
        data_to_blur=data_to_blur,
        results_detection=detect_sensitive_data_step.outputs.results_path,
    )
    blur_images_step.outputs.results_path = Output(
        type="uri_folder", mode="rw_mount", path=outputs["results_path"]
    )

    return {}


def main():
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()

    data_to_blur = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["inference_pipeline"]["inputs"]["data_to_blur"],
        description="Data to be blurred",
    )
    model = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["inference_pipeline"]["inputs"]["model"],
        description="Model to use for the blurring",
    )
    metadata_pipeline_job = inference_pipeline(data_to_blur=data_to_blur, model=model)

    metadata_pipeline_job.settings.default_compute = settings["aml_experiment_details"][
        "compute_name"
    ]
    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=metadata_pipeline_job, experiment_name="metadata_pipeline"
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()