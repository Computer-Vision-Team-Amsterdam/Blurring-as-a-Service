import json

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
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def performance_evaluation_pipeline(
    dataset_path,
    coco_annotations,
    model,
    metrics_results,
    yolo_validation_output,
    model_parameters_json,
    metrics_metadata_json,
):
    validate_model_step = validate_model(
        mounted_dataset=dataset_path,
        model=model,
        model_parameters_json=model_parameters_json,  # no outputs here
    )
    validate_model_step.outputs.yolo_validation_output = Output(
        type="uri_folder",
        mode="rw_mount",
        path=yolo_validation_output.result(),
        description="Results of the yolo run",
    )

    coco_evaluation_step = evaluate_with_coco(  # type: ignore # noqa: F841
        coco_annotations=coco_annotations,
        yolo_validation_output=validate_model_step.outputs.yolo_validation_output,
        model_parameters_json=model_parameters_json,
        metrics_metadata_json=metrics_metadata_json,
    )

    custom_evaluation_step = evaluate_with_cvt_metrics(  # type: ignore # noqa: F841
        mounted_dataset=dataset_path,
        coco_annotations=coco_annotations,
        yolo_validation_output=validate_model_step.outputs.yolo_validation_output,
        model_parameters_json=model_parameters_json,
        metrics_metadata_json=metrics_metadata_json,
    )

    custom_evaluation_step.outputs.metrics_results = Output(
        type="uri_folder",
        mode="rw_mount",
        path=metrics_results.result(),
        description="Path to store the md files from tba and fnr metrics.",
    )

    return {}


def main():
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()

    inputs = settings["performance_evaluation_pipeline"]["inputs"]
    outputs = settings["performance_evaluation_pipeline"]["outputs"]
    model_parameters = settings["performance_evaluation_pipeline"]["model_parameters"]
    model_parameters_json = json.dumps(model_parameters)
    metrics_metadata = settings["performance_evaluation_pipeline"]["metrics_metadata"]
    metrics_metadata_json = json.dumps(metrics_metadata)

    dataset_path = Input(
        type=AssetTypes.URI_FOLDER,
        path=inputs["dataset_path"],
        description="Dataset root folder. Must be in yolo format for validation. Contains images/val and labels/val",
    )

    coco_annotations = Input(
        type=AssetTypes.URI_FILE,
        path=inputs["coco_annotations"],
        description="Corresponds to metadata_pipeline['outputs']['coco_annotations']",
    )

    model = Input(
        type=AssetTypes.URI_FOLDER,
        path=inputs["model"],
        description="Model to use for the blurring",
    )

    performance_evaluation_pipeline_job = performance_evaluation_pipeline(
        dataset_path=dataset_path,
        coco_annotations=coco_annotations,
        model=model,
        metrics_results=outputs["metrics_results"],
        yolo_validation_output=outputs["yolo_validation_output"],
        model_parameters_json=model_parameters_json,
        metrics_metadata_json=metrics_metadata_json,
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
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
