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
from blurring_as_a_service.performace_evaluation_pipeline.components.validate_model import (
    validate_model,
)
from blurring_as_a_service.settings.flags import PipelineFlag
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def performance_evaluation_pipeline(
    validation_data,
    annotations_for_coco_metrics,
    yolo_yaml_path,
    yolo_validation_output,
    model,
    yolo_run_name,
    annotations_for_custom_metrics,
):
    validate_model_step = validate_model(
        mounted_dataset=validation_data, model=model, yolo_run_name=yolo_run_name
    )
    validate_model_step.outputs.yolo_validation_output = Output(
        type="uri_folder", mode="rw_mount", path=yolo_validation_output.result()
    )
    validate_model_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=yolo_yaml_path.result()
    )

    coco_evaluation_step = evaluate_with_coco(  # type: ignore # noqa: F841
        annotations_for_coco_metrics=annotations_for_coco_metrics,
        yolo_output_folder=validate_model_step.outputs.yolo_validation_output,
        yolo_run_name=yolo_run_name,
    )

    custom_evaluation_step = evaluate_with_cvt_metrics(  # type: ignore # noqa: F841
        mounted_dataset=validation_data,
        yolo_output_folder=validate_model_step.outputs.yolo_validation_output,
        annotations_for_custom_metrics=annotations_for_custom_metrics,
        yolo_run_name=yolo_run_name,
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

    annotations_for_coco_metrics = Input(
        type=AssetTypes.URI_FILE, path=inputs["annotations_for_coco_metrics"]
    )

    model = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["performance_evaluation_pipeline"]["inputs"]["model"],
        description="Model to use for the blurring",
    )

    annotations_for_custom_metrics = Input(
        type=AssetTypes.URI_FILE, path=inputs["annotations_for_custom_metrics"]
    )

    performance_evaluation_pipeline_job = performance_evaluation_pipeline(
        validation_data=validation_images_path,
        annotations_for_coco_metrics=annotations_for_coco_metrics,
        model=model,
        yolo_run_name=settings["performance_evaluation_pipeline"]["yolo_run_name"],
        yolo_validation_output=outputs["yolo_validation_output"],
        annotations_for_custom_metrics=annotations_for_custom_metrics,
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
