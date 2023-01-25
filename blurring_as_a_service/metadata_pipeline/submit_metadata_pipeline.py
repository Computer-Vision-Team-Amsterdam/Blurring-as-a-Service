import yaml
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def metadata_pipeline(input_data, output_data_path):
    from blurring_as_a_service.metadata_pipeline.components.convert_coco_to_yolo import (
        convert_coco_to_yolo,
    )

    convert_coco_to_yolo = convert_coco_to_yolo(input_data=input_data)
    convert_coco_to_yolo.outputs.output_folder = Output(
        type=AssetTypes.URI_FOLDER, path=output_data_path.result(), mode="rw_mount"
    )
    return {}


def main(input_data_path: str, output_data_path: str):
    with open("aml-config.yml", "r") as fs:
        aml_config = yaml.safe_load(fs)
        experiment_details = aml_config["experiment_details"]

    aml_interface = AMLInterface()

    submodules = ["panorama"]
    aml_interface.create_aml_environment(
        experiment_details["env_name"], submodules=submodules
    )

    input_data = Input(type=AssetTypes.URI_FILE, path=input_data_path)

    metadata_pipeline_job = metadata_pipeline(
        input_data=input_data, output_data_path=output_data_path
    )

    metadata_pipeline_job.settings.default_compute = experiment_details["compute_name"]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=metadata_pipeline_job, experiment_name="metadata_pipeline"
    )

    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    input_data_path = "azureml:coco_annotations_to_convert_to_yolo:1"
    output_data_path = (
        "azureml://subscriptions/b5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14/resourcegroups/"
        "cvo-aml-p-rg/workspaces/cvo-weu-aml-p-xnjyjutinwfyu/datastores/annotations_datastore/"
        "paths/annotations-projects/07-25-2022_120550_UTC/test_sebastian/yolo_test_4"
    )

    main(input_data_path, output_data_path)
