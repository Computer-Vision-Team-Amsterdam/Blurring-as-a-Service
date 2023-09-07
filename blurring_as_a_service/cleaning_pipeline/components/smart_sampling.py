import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="smart_sampling",
    display_name="Smart sample images from input_structured",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def smart_sampling(
    input_structured_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    customer_cvt_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to smart sample images from input_structured.
    In order to be able to re-train and evaluate the model.

    Parameters
    ----------
    input_structured_folder:
        Path of the mounted folder containing the images.
    customer_cvt_folder:
        Path of the customer data inside the CVT storage account.
    """
    # TODO: Implement the smart sampling. Ticket: BCV-52:
    # For manual inspection keep 10 images of blurred images.
    # Keep the entire sample from raw images.
    return
