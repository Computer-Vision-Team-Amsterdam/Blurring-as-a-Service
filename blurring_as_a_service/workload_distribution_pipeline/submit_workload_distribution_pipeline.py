import json

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface
from blurring_as_a_service.workload_distribution_pipeline.components.split_workload import (
    split_workload,
)


@pipeline()
def workload_distribution_pipeline(data_folder, number_of_batches):
    outputs = workload_distribution_settings["outputs"]
    date_folders = workload_distribution_settings["inputs"]["date_folders"]
    date_folders_json = json.dumps(date_folders)
    detect_sensitive_data_step = split_workload(
        data_folder=data_folder, date_folders_json=date_folders_json, number_of_batches=number_of_batches
    )
    detect_sensitive_data_step.outputs.results_folder = Output(
        type="uri_folder", mode="rw_mount", path=outputs["results_folder"]
    )

    return {}


def main():
    data_folder = Input(
        type=AssetTypes.URI_FOLDER,
        path=workload_distribution_settings["inputs"]["data_folder"],
        description="Folder containing the images",
    )
    workload_distribution_pipeline_job = workload_distribution_pipeline(
        data_folder=data_folder,
        number_of_batches=workload_distribution_settings["inputs"][
            "number_of_batches"
        ],
    )
    workload_distribution_pipeline_job.settings.default_compute = settings[
        "aml_experiment_details"
    ]["compute_name"]

    aml_interface = AMLInterface()
    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=workload_distribution_pipeline_job,
        experiment_name="workload_distribution_pipeline",
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    workload_distribution_settings = settings["workload_distribution_pipeline"]

    main()
