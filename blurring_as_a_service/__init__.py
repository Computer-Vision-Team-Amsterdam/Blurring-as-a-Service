import os

from aml_interface.aml_interface import AMLInterface

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

aml_interface = AMLInterface()

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)

settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
