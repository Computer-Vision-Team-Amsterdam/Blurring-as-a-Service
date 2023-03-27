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


class TrainingPipelineSpec(SettingsSpecModel):
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    flags: List[str] = []


class WorkloadDistributionPipelineSpec(SettingsSpecModel):
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None


class InferencePipelineSpec(SettingsSpecModel):
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    flags: List[str] = []


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

    aml_experiment_details: AMLExperimentDetailsSpec = None
    metadata_pipeline: MetadataPipelineSpec = None
    performance_evaluation_pipeline: PerformanceEvaluationPipelineSpec = None
    training_pipeline: TrainingPipelineSpec = None
    workload_distribution_pipeline: WorkloadDistributionPipelineSpec = None
    inference_pipeline: InferencePipelineSpec = None
    logging: LoggingSpec = LoggingSpec()
