from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.metadata_pipeline.components.convert_azure_coco_to_coco import (
    convert_azure_coco_to_coco,
)
from blurring_as_a_service.metadata_pipeline.components.convert_azure_coco_to_yolo import (
    convert_azure_coco_to_yolo,
)
from blurring_as_a_service.metadata_pipeline.components.create_metadata import (
    create_metadata,
)
from blurring_as_a_service.settings.flags import PipelineFlag
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def metadata_pipeline(
    coco_annotations,
    labels_path,
    images_path,
    metadata_path,
    metadata_path_coco_evaluation,
):
    metadata_flags = BlurringAsAServiceSettings.get_settings()["metadata_pipeline"][
        "flags"
    ]
    if metadata_flags & PipelineFlag.CONVERT_AZURE_COCO_TO_YOLO:
        convert_azure_coco_to_yolo_step = convert_azure_coco_to_yolo(
            input_data=coco_annotations
        )
        convert_azure_coco_to_yolo_step.outputs.output_folder = Output(
            type=AssetTypes.URI_FOLDER, path=labels_path.result(), mode="rw_mount"
        )

    if metadata_flags & PipelineFlag.CONVERT_AZURE_COCO_TO_COCO:
        convert_azure_coco_to_coco_step = convert_azure_coco_to_coco(
            input_data=coco_annotations
        )
        convert_azure_coco_to_coco_step.outputs.output_file = Output(
            type=AssetTypes.URI_FILE,
            path=metadata_path_coco_evaluation.result(),
            mode="rw_mount",
        )
    if metadata_flags & PipelineFlag.CREATE_METADATA:
        metadata_retriever_step = create_metadata(input_directory=images_path)
        metadata_retriever_step.outputs.output_file = Output(
            type=AssetTypes.URI_FILE, path=metadata_path.result(), mode="rw_mount"
        )
    return {}


def main():
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()

    inputs = settings["metadata_pipeline"]["inputs"]
    outputs = settings["metadata_pipeline"]["outputs"]
    coco_annotations = Input(type=AssetTypes.URI_FILE, path=inputs["coco_annotations"])
    images_path = Input(type=AssetTypes.URI_FOLDER, path=inputs["images_path"])

    metadata_pipeline_job = metadata_pipeline(
        coco_annotations=coco_annotations,
        labels_path=outputs["labels_path"],
        images_path=images_path,
        metadata_path=outputs["metadata_path"],
        metadata_path_coco_evaluation=outputs["metadata_path_coco_evaluation"],
    )
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
