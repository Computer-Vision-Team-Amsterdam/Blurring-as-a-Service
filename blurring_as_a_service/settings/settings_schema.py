from typing import Any, Dict, List

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AMLExperimentDetailsSpec(SettingsSpecModel):
    compute_name: str = None
    env_name: str = None
    env_version: int = None
    src_dir: str = None


class MetadataPipelineSpec(SettingsSpecModel):
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    flags: List[str] = []

    def __init__(self, inputs=None, outputs=None, flags=None, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, flags=flags, **kwargs)

        # Get paths from inputs and outputs
        images_path = self.inputs.get("images_path", "")
        yolo_annotations = self.outputs.get("yolo_annotations", "")

        # Check images_path and labels_path for the required structure
        if not images_path.endswith("images/val"):
            raise ValueError(
                "The image files must be stored in the dataset_name/images/val structure."
            )
        if not yolo_annotations.endswith("labels/val"):
            raise ValueError(
                "The yolo labels must be stored in the dataset_name/labels/val structure."
            )

        # Check if images and labels share the same root folder
        if images_path.split("/")[-3] != yolo_annotations.split("/")[-3]:
            raise ValueError(
                "The images and the labels are not under the same root folder, as expected in yolov5."
            )


class ValidationModelParameters(SettingsSpecModel):
    imgsz: int
    name: str
    save_blurred_image: bool


class MetricsMetadata(SettingsSpecModel):
    image_height: int
    image_width: int
    image_area: int


class PerformanceEvaluationPipelineSpec(SettingsSpecModel):
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
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


class WorkloadDistributionPipelineInputs(SettingsSpecModel):
    data_folder: str
    date_folders: List[str]
    number_of_batches: int


class WorkloadDistributionPipelineSpec(SettingsSpecModel):
    inputs: WorkloadDistributionPipelineInputs
    outputs: Dict[str, str] = None


class InferenceModelParameters(SettingsSpecModel):
    imgsz: int = 4000
    save_txt: bool = True
    exist_ok: bool = True
    skip_evaluation: bool = True
    save_blurred_image: bool = True
    batch_size: int = 1


class InferenceCustomerPipelineSpec(SettingsSpecModel):
    customer_name: str
    model_parameters: InferenceModelParameters

class MoveDataSpec(SettingsSpecModel):
    customers: List[str] = None


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


class BlurringAsAServiceSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    aml_experiment_details: AMLExperimentDetailsSpec
    metadata_pipeline: MetadataPipelineSpec = None
    performance_evaluation_pipeline: PerformanceEvaluationPipelineSpec = None
    training_pipeline: TrainingPipelineSpec = None
    workload_distribution_pipeline: WorkloadDistributionPipelineSpec = None
    inference_pipeline: InferenceCustomerPipelineSpec = None
    move_data_pipeline: MoveDataSpec = None
    logging: LoggingSpec = LoggingSpec()
