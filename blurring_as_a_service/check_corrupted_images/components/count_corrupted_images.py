import os
import sys
import logging

from azure.ai.ml.constants import AssetTypes
from mldesigner import Output, command_component


sys.path.append("../../..")
from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
BlurringAsAServiceSettings.set_from_yaml(config_path)
settings = BlurringAsAServiceSettings.get_settings()
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from blurring_as_a_service.check_corrupted_images.source.count_corrupted_images_per_folder import count_corrupted_images_per_folder  # noqa: E402


aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="count_corrupted_images",
    display_name="Count corrupted images in input_structured folder",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def count_corrupted_images(
    input_structured_container: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    image_counts = count_corrupted_images_per_folder(input_structured_container)
    for folder, count in image_counts.items():
        logging.info(f"{folder}: {count} images")
