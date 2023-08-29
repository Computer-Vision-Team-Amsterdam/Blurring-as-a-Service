import json
import os
from datetime import datetime

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
def performance_evaluation_pipeline():
    datastore_name = performance_evaluation_settings["datastore"]
    inputs = performance_evaluation_settings["inputs"]
    model_parameters = performance_evaluation_settings["model_parameters"]
    model_parameters_json = json.dumps(model_parameters)
    metrics_metadata = performance_evaluation_settings["metrics_metadata"]
    metrics_metadata_json = json.dumps(metrics_metadata)

    execution_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_folder = f"performance_evaluation_pipeline/{execution_time}_{model_parameters_json}_{metrics_metadata_json}"

    yolo_dataset_path = os.path.join(
        aml_interface.get_azureml_path(datastore_name=datastore_name),
        inputs["yolo_dataset"],
    )

    coco_annotations_path = os.path.join(
        aml_interface.get_azureml_path(datastore_name=datastore_name),
        inputs["coco_annotations"],
    )

    yolo_validation_output_path = os.path.join(
        aml_interface.get_azureml_path(datastore_name=datastore_name),
        base_output_folder,
    )

    cvt_metrics_path = os.path.join(
        aml_interface.get_azureml_path(datastore_name=datastore_name),
        base_output_folder,
        "cvt_metrics",
    )
    yolo_dataset = Input(
        type=AssetTypes.URI_FOLDER,
        path=yolo_dataset_path,
        description="Dataset root folder. Must be in yolo format for validation. Contains images/val and labels/val",
    )

    coco_annotations = Input(
        type=AssetTypes.URI_FILE,
        path=coco_annotations_path,
        description="Corresponds to metadata_pipeline['outputs']['coco_annotations']",
    )

    model = Input(
        type=AssetTypes.CUSTOM_MODEL,
        path=f"azureml:{inputs['model_name']}:{inputs['model_version']}",
        description="Model weights for evaluation",
    )

    validate_model_step = validate_model(
        mounted_dataset=yolo_dataset,
        model=model,
        model_parameters_json=model_parameters_json,  # no outputs when calling command_components
    )
    validate_model_step.outputs.yolo_validation_output = Output(
        type="uri_folder",
        mode="rw_mount",
        path=yolo_validation_output_path,
        description="Results of the yolo run",
    )

    coco_evaluation_step = evaluate_with_coco(  # type: ignore # noqa: F841
        coco_annotations=coco_annotations,
        yolo_validation_output=validate_model_step.outputs.yolo_validation_output,
        model_parameters_json=model_parameters_json,
        metrics_metadata_json=metrics_metadata_json,
    )

    custom_evaluation_step = evaluate_with_cvt_metrics(  # type: ignore # noqa: F841
        mounted_dataset=yolo_dataset,
        coco_annotations=coco_annotations,
        yolo_validation_output=validate_model_step.outputs.yolo_validation_output,
        model_parameters_json=model_parameters_json,
        metrics_metadata_json=metrics_metadata_json,
    )

    custom_evaluation_step.outputs.metrics_results = Output(
        type="uri_folder",
        mode="rw_mount",
        path=cvt_metrics_path,
        description="Where to store the md files from tba and fnr metrics.",
    )

    return {}


def main():
    performance_evaluation_pipeline_job = performance_evaluation_pipeline()
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
    settings = BlurringAsAServiceSettings.get_settings()
    performance_evaluation_settings = settings["performance_evaluation_pipeline"]

    aml_interface = AMLInterface()

    main()
