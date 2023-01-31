from typing import Dict

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.metadata_pipeline.components.convert_coco_to_yolo import (
    convert_coco_to_yolo,
)
from blurring_as_a_service.metadata_pipeline.components.create_metadata import (
    create_metadata,
)
from blurring_as_a_service.settings.flags import PipelineFlag
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def metadata_pipeline(coco_annotations, labels_path, images_path, metadata_path):
    metadata_flags = BlurringAsAServiceSettings.get_settings()["metadata_pipeline"][
        "flags"
    ]
    if metadata_flags & PipelineFlag.CONVERT_COCO_TO_YOLO:
        convert_coco_to_yolo_step = convert_coco_to_yolo(input_data=coco_annotations)
        convert_coco_to_yolo_step.outputs.output_folder = Output(
            type=AssetTypes.URI_FOLDER, path=labels_path.result(), mode="rw_mount"
        )
    if metadata_flags & PipelineFlag.CREATE_METADATA:
        metadata_retriever_step = create_metadata(input_directory=images_path)
        metadata_retriever_step.outputs.output_file = Output(
            type=AssetTypes.URI_FILE, path=metadata_path.result(), mode="rw_mount"
        )
    return {}


def main(inputs: Dict[str, str], outputs: Dict[str, str]):
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()
    if settings["metadata_pipeline"]["flags"] & PipelineFlag.CREATE_ENVIRONMENT:
        custom_packages = {
            "panorama": "git+https://github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.2",
        }
        aml_interface.create_aml_environment(
            settings["aml_experiment_details"]["env_name"],
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

    metadata_pipeline_job.settings.default_compute = settings["aml_experiment_details"][
        "compute_name"
    ]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=metadata_pipeline_job, experiment_name="metadata_pipeline"
    )

    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    settings = BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main(
        settings["metadata_pipeline"]["inputs"],
        settings["metadata_pipeline"]["outputs"],
    )
