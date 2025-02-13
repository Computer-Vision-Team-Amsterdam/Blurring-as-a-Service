import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

from blurring_as_a_service import settings

sys.path.append("../../..")
from blurring_as_a_service.metadata_pipeline.source.metadata_retriever import (  # noqa: E402
    MetadataRetriever,
)

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="create_metadata",
    display_name="Create metadata",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def create_metadata(
    input_directory: Input(type=AssetTypes.URI_FOLDER), metadata_path: Output(type=AssetTypes.URI_FILE)  # type: ignore # noqa: F821
):
    """
    Pipeline step to create metadata file from panorama images. Only works for panorama ids that exist in the API.

    Parameters
    ----------
    input_directory: panorama images
    metadata_path: json with images metadata

    Returns
    -------

    """
    MetadataRetriever(
        images_directory_path=input_directory
    ).generate_and_store_metadata(metadata_path)
