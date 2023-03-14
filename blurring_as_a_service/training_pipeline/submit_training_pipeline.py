from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.training_pipeline.components.train_model import train_model
from blurring_as_a_service.utils.aml_interface import AMLInterface


@pipeline()
def training_pipeline(training_data, model_weights, trained_model):
    train_model_step = train_model(
        mounted_dataset=training_data, model_weights=model_weights
    )
    train_model_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=training_data.path
    )
    train_model_step.outputs.trained_model = Output(
        type="uri_folder", mode="rw_mount", path=trained_model.result()
    )
    return {}


def main():
    aml_interface = AMLInterface()
    settings = BlurringAsAServiceSettings.get_settings()

    training_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["training_pipeline"]["inputs"]["training_data"],
    )
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["training_pipeline"]["inputs"]["model_weights"],
    )
    training_pipeline_job = training_pipeline(
        training_data=training_data,
        model_weights=model_weights,
        trained_model=settings["training_pipeline"]["outputs"]["trained_model"],
    )
    training_pipeline_job.settings.default_compute = settings["aml_experiment_details"][
        "compute_name"
    ]

    pipeline_job = aml_interface.submit_pipeline_job(
        pipeline_job=training_pipeline_job, experiment_name="metadata_pipeline"
    )
    aml_interface.wait_until_job_completes(pipeline_job.name)


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
