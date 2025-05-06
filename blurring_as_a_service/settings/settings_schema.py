from typing import Dict, List

from yolo_model_development_kit.settings.settings_schema import (
    AMLExperimentDetailsSpec,
    InferencePipelineSpec,
    LoggingSpec,
    SettingsSpecModel,
)


class MetadataPipelineSpec(SettingsSpecModel):
    datastore: str
    tagged_data: bool
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    flags: List[str] = []

    def __init__(self, datastore=None, inputs=None, outputs=None, flags=None, **kwargs):
        super().__init__(
            datastore=datastore, inputs=inputs, outputs=outputs, flags=flags, **kwargs
        )

        # Get paths from inputs and outputs
        images = self.inputs.get("images", "")
        yolo_annotations = self.outputs.get("yolo_annotations", "")

        # Check images and yolo_annotations for the required structure
        if not images.endswith("images/val"):
            raise ValueError(
                "The image files must be stored in the dataset_name/images/val structure."
            )
        if not yolo_annotations.endswith("labels/val"):
            raise ValueError(
                "The yolo labels must be stored in the dataset_name/labels/val structure."
            )

        # Check if images and labels share the same root folder
        if images.split("/")[0] != yolo_annotations.split("/")[0]:
            raise ValueError(
                "The images and the labels are not under the same root folder, as expected in yolov5."
            )


class PreInferencePipelineInputs(SettingsSpecModel):
    number_of_batches: int


class PreInferencePipelineSpec(SettingsSpecModel):
    datastore_input: str
    datastore_input_path: str
    datastore_output: str
    inputs: PreInferencePipelineInputs


class DatabaseCredentialsSpec(SettingsSpecModel):
    db_username: str
    db_hostname: str
    db_name: str
    client_id: str


class BaaSInferencePipelineSpec(InferencePipelineSpec):
    database_parameters: DatabaseCredentialsSpec


class SmartSamplingPipelineSpec(SettingsSpecModel):
    quality_check_sample_size: int = 10
    conf_score_threshold: float = 0.0005
    sampling_ratio: float = 0.5


class APIEndpointSpec(SettingsSpecModel):
    endpoint_name: str
    deployment_color: str
    instance_type: str
    model_name: str
    model_version: str


class BlurringAsAServiceSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    metadata_pipeline: MetadataPipelineSpec = None
    pre_inference_pipeline: PreInferencePipelineSpec = None
    inference_pipeline: BaaSInferencePipelineSpec = None
    sampling_parameters: SmartSamplingPipelineSpec = None
    api_endpoint: APIEndpointSpec = None
    logging: LoggingSpec = LoggingSpec()
