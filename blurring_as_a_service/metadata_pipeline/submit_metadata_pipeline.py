from typing import Dict

import yaml
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.metadata_pipeline.components.convert_coco_to_yolo import (
    convert_coco_to_yolo,
)
from blurring_as_a_service.metadata_pipeline.components.create_metadata import (
    create_metadata,
)
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def metadata_pipeline(coco_annotations, labels_path, images_path, metadata_path):
    convert_coco_to_yolo_step = convert_coco_to_yolo(input_data=coco_annotations)
    convert_coco_to_yolo_step.outputs.output_folder = Output(
        type=AssetTypes.URI_FOLDER, path=labels_path.result(), mode="rw_mount"
    )
    metadata_retriever_step = create_metadata(input_directory=images_path)
    metadata_retriever_step.outputs.output_file = Output(
        type=AssetTypes.URI_FILE, path=metadata_path.result(), mode="rw_mount"
    )
    return {}


def main(inputs: Dict[str, str], outputs: Dict[str, str]):
    with open("aml-config.yml", "r") as fs:
        aml_config = yaml.safe_load(fs)
        experiment_details = aml_config["experiment_details"]

    aml_interface = AMLInterface()

    custom_packages = {
        "panorama": "git+https://github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.2",
    }
    aml_interface.create_aml_environment(
        experiment_details["env_name"],
        project_name="blurring-as-a-service",
        custom_packages=custom_packages,
    )

    coco_annotations = Input(type=AssetTypes.URI_FILE, path=inputs["coco_annotations"])
    images_path = Input(type=AssetTypes.URI_FOLDER, path=inputs["images_path"])

    metadata_pipeline_job = metadata_pipeline(
        coco_annotations=coco_annotations,
        labels_path=outputs["labels_path"],
        images_path=images_path,
        metadata_path=outputs["metadata_path"],
    )

    metadata_pipeline_job.settings.default_compute = experiment_details["compute_name"]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=metadata_pipeline_job, experiment_name="metadata_pipeline"
    )

    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    inputs = {
        "coco_annotations": "azureml:coco_annotations_to_convert_to_yolo:1",
        "images_path": (
            "azureml://subscriptions/b5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14/resourcegroups/"
            "cvo-aml-p-rg/workspaces/cvo-weu-aml-p-xnjyjutinwfyu/datastores/annotations_datastore/"
            "paths/annotations-projects/07-25-2022_120550_UTC/test_sebastian/sample/images"
        ),
    }
    outputs = {
        "labels_path": (
            "azureml://subscriptions/b5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14/resourcegroups/"
            "cvo-aml-p-rg/workspaces/cvo-weu-aml-p-xnjyjutinwfyu/datastores/annotations_datastore/"
            "paths/annotations-projects/07-25-2022_120550_UTC/test_sebastian/sample/labels"
        ),
        "metadata_path": (
            "azureml://subscriptions/b5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14/resourcegroups/"
            "cvo-aml-p-rg/workspaces/cvo-weu-aml-p-xnjyjutinwfyu/datastores/annotations_datastore/"
            "paths/annotations-projects/07-25-2022_120550_UTC/test_sebastian/sample/metadata/metadata.json"
        ),
    }

    main(inputs, outputs)
