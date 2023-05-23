import os
import sys

from azure.ai.ml import Output
from azure.ai.ml.dsl import pipeline

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from blurring_as_a_service.inference_pipeline.components.move_files import (  # noqa: E402
    move_files,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)
from blurring_as_a_service.utils.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def move_files_pipeline(settings):
    customer_list = settings["inputs"]["customers"]

    for customer in customer_list:
        move_data = move_files()

        azureml_input = settings["inputs"]["container_root"]
        azureml_output = settings["outputs"]["container_root"]

        azureml_input_formatted = azureml_input.format(
            # subscription="your_subscription",  # TODO get from config.json
            # resourcegroup="your_resource_group",
            # workspace="your_workspace",
            datastore_name=customer
        )

        azureml_output_formatted = azureml_output.format(
            # subscription="your_subscription",
            # resourcegroup="your_resource_group",
            # workspace="your_workspace",
            datastore_name=customer
        )

        # NOTE We need to use Output to also delete the files.
        move_data.outputs.input_container = Output(
            type="uri_folder", mode="rw_mount", path=azureml_input_formatted
        )

        move_data.outputs.output_container = Output(
            type="uri_folder", mode="rw_mount", path=azureml_output_formatted
        )

    return {}


def main():
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()

    inference_pipeline_job = move_files_pipeline(settings["move_data_pipeline"])

    inference_pipeline_job.settings.default_compute = settings[
        "aml_experiment_details"
    ]["compute_name"]

    aml_interface = AMLInterface()
    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=inference_pipeline_job, experiment_name="move_data_pipeline"
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    main()
