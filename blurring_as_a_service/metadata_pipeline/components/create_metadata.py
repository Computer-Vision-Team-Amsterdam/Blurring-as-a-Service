import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.metadata_pipeline.source.metadata_retriever import (  # noqa: E402
    MetadataRetriever,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


@command_component(
    name="create_metadata",
    display_name="Create metadata",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def create_metadata(
    input_directory: Input(type=AssetTypes.URI_FOLDER), output_file: Output(type=AssetTypes.URI_FILE)  # type: ignore # noqa: F821
):
    MetadataRetriever(
        images_directory_path=input_directory
    ).generate_and_store_metadata(output_file)
