from cvtoolkit.settings.settings_helper import GenericSettings, Settings, strings2flags
from pydantic import BaseModel

from blurring_as_a_service.settings.flags import PipelineFlag
from blurring_as_a_service.settings.settings_schema import (
    BlurringAsAServiceSettingsSpec,
)


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
        print("BlurringAsAServiceSettings.set_from_yaml: filename = ", filename)
        return super().set_from_yaml(filename, spec)
