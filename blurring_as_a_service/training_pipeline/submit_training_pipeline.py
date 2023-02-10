from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.training_pipeline.components.train_model import train_model
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def training_pipeline(training_data):
    train_model_step = train_model(mounted_dataset=training_data)
    train_model_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=training_data.path
    )
    return {}


def main():
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()

    # TODO: Impossible to create an env for the training with the SDKv2 at the moment.
    # This needs to be uncommented and fixed when it becomes possible:
    # if settings["training_pipeline"]["flags"] & PipelineFlag.CREATE_ENVIRONMENT:
    #     custom_packages = {
    #         "panorama": "git+https://github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.2",
    #     }
    #     aml_interface.create_aml_environment(
    #         settings["aml_experiment_details"]["env_name"],
    #         project_name="blurring-as-a-service",
    #         custom_packages=custom_packages,
    #     )

    training_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["training_pipeline"]["inputs"]["training_data"],
    )
    metadata_pipeline_job = training_pipeline(training_data=training_data)
    metadata_pipeline_job.settings.default_compute = settings["aml_experiment_details"][
        "compute_name"
    ]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=metadata_pipeline_job, experiment_name="metadata_pipeline"
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
