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


class PerformanceEvaluationPipelineSpec(SettingsSpecModel):
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    yolo_run_name: str = None
    flags: List[str] = []


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
