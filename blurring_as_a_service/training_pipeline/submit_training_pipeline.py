from aml_interface.aml_interface import AMLInterface  # noqa: E402
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.training_pipeline.components.train_model import (  # noqa: E402
    train_model,
)

BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()


@pipeline()
def training_pipeline():
    trained_model = settings["training_pipeline"]["outputs"]["trained_model"]
    training_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["training_pipeline"]["inputs"]["training_data"],
    )
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=settings["training_pipeline"]["inputs"]["model_weights"],
    )
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


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        training_pipeline, "training_pipeline", default_compute
    )
