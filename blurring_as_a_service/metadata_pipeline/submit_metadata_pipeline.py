import os
from datetime import datetime

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
def metadata_pipeline():
    execution_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_folder = f"metadata_pipeline/{execution_time}"

    datastore_name = metadata_settings["datastore"]
    inputs = metadata_settings["inputs"]
    outputs = metadata_settings["outputs"]
    metadata_flags = metadata_settings["flags"]

    coco_annotations_in_path = os.path.join(
        aml_interface.get_datastore_full_path(datastore_name=datastore_name),
        inputs["coco_annotations"],
    )

    yolo_annotations_path = os.path.join(
        aml_interface.get_datastore_full_path(datastore_name=datastore_name),
        outputs["yolo_annotations"],
    )

    images_path = os.path.join(
        aml_interface.get_datastore_full_path(datastore_name=datastore_name),
        inputs["images"],
    )

    coco_annotations_out_path = os.path.join(
        aml_interface.get_datastore_full_path(datastore_name=datastore_name),
        base_output_folder,
        outputs["coco_annotations"],
    )

    metadata_path = os.path.join(
        aml_interface.get_datastore_full_path(datastore_name=datastore_name),
        base_output_folder,
        outputs["metadata"],
    )

    coco_annotations_in = Input(
        type=AssetTypes.URI_FILE,
        path=coco_annotations_in_path,
        description="Path to Data Labeling annotation json file. This is in incorrect/non-standard coco format.",
    )

    images = Input(
        type=AssetTypes.URI_FOLDER,
        path=images_path,
        description="Path to images folder that correspond to the Data Labeling annotations.",
    )

    if metadata_flags & PipelineFlag.CONVERT_AZURE_COCO_TO_YOLO:
        convert_azure_coco_to_yolo_step = convert_azure_coco_to_yolo(
            coco_annotations_in=coco_annotations_in,
            tagged_data=metadata_settings["tagged_data"],
        )

        convert_azure_coco_to_yolo_step.outputs.yolo_annotations = Output(
            type=AssetTypes.URI_FOLDER,
            path=yolo_annotations_path,
            mode="rw_mount",
            description="Yolo annotations",
        )

    if metadata_flags & PipelineFlag.CONVERT_AZURE_COCO_TO_COCO:
        convert_azure_coco_to_coco_step = convert_azure_coco_to_coco(
            coco_annotations_in=coco_annotations_in,
            image_width=outputs["image_width"],
            image_height=outputs["image_height"],
        )
        convert_azure_coco_to_coco_step.outputs.coco_annotations_out = Output(
            type=AssetTypes.URI_FILE,
            path=coco_annotations_out_path,
            mode="rw_mount",
            description="Standard coco annotation file",
        )

    if metadata_flags & PipelineFlag.CREATE_METADATA:
        metadata_retriever_step = create_metadata(input_directory=images)
        metadata_retriever_step.outputs.metadata_path = Output(
            type=AssetTypes.URI_FILE,
            path=metadata_path,
            mode="rw_mount",
            description="Json file with images metadata gotten via the panorama API.",
        )
    return {}


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    metadata_settings = settings["metadata_pipeline"]

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        metadata_pipeline, "metadata_pipeline", default_compute
    )
