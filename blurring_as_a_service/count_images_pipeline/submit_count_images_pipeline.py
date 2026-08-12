import os

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from aml_interface.aml_interface import AMLInterface  # noqa: E402

from blurring_as_a_service.count_images_pipeline.components.count_images import (  # noqa: E402
    count_images,
)


@pipeline()
def count_images_pipeline():
    aml_interface = AMLInterface()

    input_datastore_fullpath = aml_interface.get_datastore_full_path(
        settings["pre_inference_pipeline"]["inputs"]["datastore"]
    )
    input_folder = Input(
        type=AssetTypes.URI_FOLDER,
        path=os.path.join(
            input_datastore_fullpath,
            settings["pre_inference_pipeline"]["inputs"]["input_rel_path"],
        ),
        description="Input folder",
    )

    count_images(input_folder=input_folder)

    return {}


def main():
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        pipeline_function=count_images_pipeline,
        experiment_name=settings["aml_experiment_details"]["experiment_name"],
        default_compute=settings["aml_experiment_details"]["compute_name"],
        show_log=False,
    )


if __name__ == "__main__":
    main()
