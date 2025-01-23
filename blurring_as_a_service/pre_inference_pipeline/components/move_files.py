import logging
import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Output, command_component

sys.path.append("../../..")
from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402
from cvtoolkit.helpers.file_helpers import copy_file, delete_file  # noqa: E402

from blurring_as_a_service.pre_inference_pipeline.source.image_paths import (  # noqa: E402
    get_image_paths,
)
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

aml_experiment_settings = settings["aml_experiment_details"]

logger = logging.getLogger("move_files")


@command_component(
    name="move_files",
    display_name="Move files",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def move_files(
    execution_time: str,
    input_container: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_container: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to detect the areas to blur.

    Parameters
    ----------
    execution_time:
        Datetime containing when the job was executed. Used to name the folder.
    input_container:
        Path of the mounted root folder containing the images.
    output_container:
        Where to store the results in a Blob Container.

    """
    image_paths = get_image_paths(input_container)

    if not image_paths:
        logger.info("No files in the input zone. Aborting...")
        return

    target_folder_path = os.path.join(output_container, execution_time)
    os.makedirs(target_folder_path, exist_ok=True)

    for source_image_path, relative_image_path in image_paths:
        target_file_path = os.path.join(target_folder_path, relative_image_path)
        copy_file("", source_image_path, target_file_path)
        delete_file(source_image_path)

    logger.info("Files moved and removed successfully.")
