import logging

from pydantic import BaseModel

from blurring_as_a_service.settings.flags import PipelineFlag
from blurring_as_a_service.settings.settings_helper import (
    GenericSettings,
    Settings,
    strings2flags,
)
from blurring_as_a_service.settings.settings_schema import (
    BlurringAsAServiceSettingsSpec,
)

logger = logging.getLogger(__name__)


class BlurringAsAServiceSettings(Settings):  # type: ignore
    @classmethod
    def process_value(cls, k, v):
        v = super().process_value(k, v)
        if k == "flags":
            v = strings2flags(v, PipelineFlag)
        return v

    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = BlurringAsAServiceSettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)
