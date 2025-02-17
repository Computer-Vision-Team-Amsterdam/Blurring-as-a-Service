from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402
from azure.ai.ml import Output
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

from blurring_as_a_service.check_corrupted_images.components.count_corrupted_images import (  # noqa: E402
    count_corrupted_images,
)


@pipeline()
def check_corrupted_images_pipeline():
    aml_interface = AMLInterface()
    azureml_output_formatted = aml_interface.get_datastore_full_path(
        settings["inference_pipeline"]["datastore_input_structured"]
    )

    count_corrupted_images_step = count_corrupted_images()

    count_corrupted_images_step.outputs.input_structured_container = Output(
        type="uri_folder", mode="rw_mount", path=azureml_output_formatted
    )

    return {}


def main():
    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        check_corrupted_images_pipeline, "count_corrupted_images", default_compute
    )


if __name__ == "__main__":
    main()
