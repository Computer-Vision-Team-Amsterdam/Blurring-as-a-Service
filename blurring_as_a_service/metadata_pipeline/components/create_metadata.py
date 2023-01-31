import sys

from mldesigner import Input, Output, command_component

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

sys.path.append("../../..")
from blurring_as_a_service.metadata_pipeline.utils.metadata_retriever import (  # noqa: E402
    MetadataRetriever,
)

aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml("config.yml")[
    "aml_experiment_details"
]


@command_component(
    name="create_metadata",
    display_name="Create metadata",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def create_metadata(
    input_directory: Input(type="uri_folder"), output_file: Output(type="uri_file")  # type: ignore # noqa: F821
):
    MetadataRetriever(
        images_directory_path=input_directory
    ).generate_and_store_metadata(output_file)
