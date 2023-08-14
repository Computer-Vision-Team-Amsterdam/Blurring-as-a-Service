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
    coco_annotations_in,
    yolo_annotations,
    images_path,
    metadata_path,
    coco_annotations_out,
):
    metadata_flags = BlurringAsAServiceSettings.get_settings()["metadata_pipeline"][
        "flags"
    ]
    if metadata_flags & PipelineFlag.CONVERT_AZURE_COCO_TO_YOLO:
        convert_azure_coco_to_yolo_step = convert_azure_coco_to_yolo(
            coco_annotations_in=coco_annotations_in
        )

        convert_azure_coco_to_yolo_step.outputs.yolo_annotations = Output(
            type=AssetTypes.URI_FOLDER, path=yolo_annotations.result(), mode="rw_mount"
        )

    if metadata_flags & PipelineFlag.CONVERT_AZURE_COCO_TO_COCO:
        convert_azure_coco_to_coco_step = convert_azure_coco_to_coco(
            coco_annotations_in=coco_annotations_in
        )
        convert_azure_coco_to_coco_step.outputs.coco_annotations_out = Output(
            type=AssetTypes.URI_FILE,
            path=coco_annotations_out.result(),
            mode="rw_mount",
        )
    if metadata_flags & PipelineFlag.CREATE_METADATA:
        metadata_retriever_step = create_metadata(input_directory=images_path)
        metadata_retriever_step.outputs.metadata_path = Output(
            type=AssetTypes.URI_FILE, path=metadata_path.result(), mode="rw_mount"
        )
    return {}


def main():
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()

    inputs = settings["metadata_pipeline"]["inputs"]
    outputs = settings["metadata_pipeline"]["outputs"]
    coco_annotations_in = Input(
        type=AssetTypes.URI_FILE, path=inputs["coco_annotations"]
    )
    images_path = Input(type=AssetTypes.URI_FOLDER, path=inputs["images_path"])

    metadata_pipeline_job = metadata_pipeline(
        coco_annotations_in=coco_annotations_in,
        yolo_annotations=outputs["yolo_annotations"],
        images_path=images_path,
        metadata_path=outputs["metadata_path"],
        coco_annotations_out=outputs["coco_annotations"],
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
