import sys

from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.metadata_pipeline.utils.metadata_retriever import (  # noqa: E402
    MetadataRetriever,
)


@command_component(
    name="create_metadata",
    display_name="Create metadata",
    environment="azureml:test-sebastian-env:87",
    code="../../../",
)
def create_metadata(
    input_directory: Input(type="uri_folder"), output_file: Output(type="uri_file")  # type: ignore # noqa: F821
):
    MetadataRetriever(
        images_directory_path=input_directory
    ).generate_and_store_metadata(output_file)
