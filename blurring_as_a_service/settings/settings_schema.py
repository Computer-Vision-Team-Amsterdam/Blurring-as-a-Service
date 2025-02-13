from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AMLExperimentDetailsSpec(SettingsSpecModel):
    compute_name: str = None
    env_name: str = None
    env_version: int = None
    src_dir: str = None
    ai_instrumentation_key: str = None


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


class ValidationModelParameters(SettingsSpecModel):
    imgsz: int
    name: str
    save_blurred_image: bool
    conf_thres: float
    no_inverted_colors: bool


class MetricsMetadata(SettingsSpecModel):
    image_height: int
    image_width: int
    image_area: int


class PerformanceEvaluationPipelineSpec(SettingsSpecModel):
    datastore: str = None
    inputs: Dict[str, str] = None
    metrics_metadata: MetricsMetadata
    model_parameters: ValidationModelParameters


class TrainingModelParameters(SettingsSpecModel):
    img_size: int = 2048
    batch_size: int = 8
    epochs: int = 2


class TrainingPipelineSpec(SettingsSpecModel):
    model_parameters: TrainingModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    flags: List[str] = []


class PreInferencePipelineInputs(SettingsSpecModel):
    number_of_batches: int


class PreInferencePipelineSpec(SettingsSpecModel):
    datastore_input: str
    datastore_input_path: str
    execution_time: str
    datastore_output: str
    inputs: PreInferencePipelineInputs


class InferenceModelParameters(SettingsSpecModel):
    batch_size: int = 1
    img_size: int = 640
    conf: float = 0.5
    save_img_flag: bool = False
    save_txt_flag: bool = False
    save_conf_flag: bool = False


class DatabaseCredentialsSpec(SettingsSpecModel):
    db_username: str
    db_hostname: str
    db_name: str
    client_id: str


class InferencePipelineSpec(SettingsSpecModel):
    model_params: InferenceModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    target_classes: List[int] = None
    sensitive_classes: List[int] = []
    target_classes_conf: Optional[float] = None
    sensitive_classes_conf: Optional[float] = None
    output_image_size: Optional[Tuple[int, int]] = None
    save_detection_images: bool = False
    save_detection_labels: bool = True
    save_all_images: bool = True
    database_parameters: DatabaseCredentialsSpec


class SmartSamplingPipelineSpec(SettingsSpecModel):
    quality_check_sample_size: int = 10
    conf_score_threshold: float = 0.0005
    sampling_ratio: float = 0.5


class LoggingSpec(SettingsSpecModel):
    loglevel_own: str = "INFO"
    own_packages: List[str] = [
        "__main__",
        "blurring_as_a_service",
    ]
    extra_loglevels: Dict[str, str] = {}
    basic_config: Dict[str, Any] = {
        "level": "WARNING",
        "format": "%(asctime)s|%(levelname)-8s|%(name)s|%(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    ai_instrumentation_key: str = ""


class BlurringAsAServiceSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    metadata_pipeline: MetadataPipelineSpec = None
    performance_evaluation_pipeline: PerformanceEvaluationPipelineSpec = None
    training_pipeline: TrainingPipelineSpec = None
    pre_inference_pipeline: PreInferencePipelineSpec = None
    inference_pipeline: InferencePipelineSpec = None
    sampling_parameters: SmartSamplingPipelineSpec = None
    logging: LoggingSpec = LoggingSpec()
