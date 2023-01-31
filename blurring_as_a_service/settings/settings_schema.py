from typing import Any, Dict, List

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class MetadataPipelineSpec(SettingsSpecModel):
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

    metadata_pipeline: MetadataPipelineSpec = None
    logging: LoggingSpec = LoggingSpec()
