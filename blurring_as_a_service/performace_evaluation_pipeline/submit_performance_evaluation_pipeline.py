from typing import Dict

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.performace_evaluation_pipeline.components.get_data import (
    get_data,
)
from blurring_as_a_service.settings.flags import PipelineFlag
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def performance_evaluation_pipeline(
    validation_images_path, txt_validation_images_names_path
):
    peformance_evaluation_flags = BlurringAsAServiceSettings.get_settings()[
        "performance_evaluation_pipeline"
    ]["flags"]
    if peformance_evaluation_flags & PipelineFlag.GET_DATA:
        get_data_step = get_data(input_folder=validation_images_path)

        get_data_step.outputs.output_file = Output(
            type=AssetTypes.URI_FILE,
            path=txt_validation_images_names_path.result(),
            mode="rw_mount",
        )

    return {}


def main(inputs: Dict[str, str], outputs: Dict[str, str]):
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()

    if (
        settings["performance_evaluation_pipeline"]["flags"]
        & PipelineFlag.CREATE_ENVIRONMENT
    ):
        custom_packages = {
            "panorama": "git+https://github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.2",
        }
        aml_interface.create_aml_environment(
            settings["aml_experiment_details"]["env_name"],
            project_name="blurring-as-a-service",
            custom_packages=custom_packages,
        )

    validation_images_path = Input(
        type=AssetTypes.URI_FOLDER, path=inputs["validation_images_path"]
    )

    performance_evaluation_pipeline_job = performance_evaluation_pipeline(
        validation_images_path=validation_images_path,
        txt_validation_images_names_path=outputs["txt_validation_images_names_path"],
    )

    performance_evaluation_pipeline_job.settings.default_compute = settings[
        "aml_experiment_details"
    ]["compute_name"]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=performance_evaluation_pipeline_job,
        experiment_name="performance_evaluation_pipeline",
    )

    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    settings = BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main(
        settings["performance_evaluation_pipeline"]["inputs"],
        settings["performance_evaluation_pipeline"]["outputs"],
    )
