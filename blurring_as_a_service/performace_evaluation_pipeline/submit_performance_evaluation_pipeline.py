from typing import Dict

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.performace_evaluation_pipeline.components.evaluate_with_coco import (
    evaluate_with_coco,
)
from blurring_as_a_service.performace_evaluation_pipeline.components.evaluate_with_cvt_metrics import (
    evaluate_with_cvt_metrics,
)
from blurring_as_a_service.performace_evaluation_pipeline.components.get_data import (
    get_data,
)
from blurring_as_a_service.performace_evaluation_pipeline.components.validate_model import (
    validate_model,
)
from blurring_as_a_service.settings.flags import PipelineFlag
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def performance_evaluation_pipeline(
    validation_data, annotations_json, yolo_yaml_path, yolo_validation_output
):
    performance_evaluation_flags = BlurringAsAServiceSettings.get_settings()[
        "performance_evaluation_pipeline"
    ]["flags"]
    if performance_evaluation_flags & PipelineFlag.GET_DATA:
        get_data_step = get_data(input_folder=validation_data)

        get_data_step.outputs.output_file = Output(
            type=AssetTypes.URI_FILE,
            path=yolo_validation_output.result(),
            mode="rw_mount",
        )
    if performance_evaluation_flags & PipelineFlag.VALIDATE_MODEL:
        validate_model_step = validate_model(mounted_dataset=validation_data)
        validate_model_step.outputs.yolo_validation_output = Output(
            type="uri_folder", mode="rw_mount", path=yolo_validation_output.result()
        )
        validate_model_step.outputs.yolo_yaml_path = Output(
            type="uri_folder", mode="rw_mount", path=yolo_yaml_path.result()
        )
        coco_evaluation_step = evaluate_with_coco(  # type: ignore # noqa: F841
            annotations_json=annotations_json,
            yolo_output_folder=validate_model_step.outputs.yolo_validation_output,
        )

        custom_evaluation_step = evaluate_with_cvt_metrics(  # type: ignore # noqa: F841
            mounted_dataset=validation_data,
            yolo_output_folder=validate_model_step.outputs.yolo_validation_output,
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

    annotations_json = Input(type=AssetTypes.URI_FILE, path=inputs["annotations_json"])

    performance_evaluation_pipeline_job = performance_evaluation_pipeline(
        validation_data=validation_images_path,
        annotations_json=annotations_json,
        yolo_validation_output=outputs["yolo_validation_output"],
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
