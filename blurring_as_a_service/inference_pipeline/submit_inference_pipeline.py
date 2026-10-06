import argparse
import os

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from aml_interface.aml_interface import AMLInterface  # noqa: E402

from blurring_as_a_service.inference_pipeline.components.detect_and_blur_sensitive_data import (  # noqa: E402
    detect_and_blur_sensitive_data,
)


@pipeline()
def inference_pipeline():
    input_datastore_fullpath = aml_interface.get_datastore_full_path(
        settings["inference_pipeline"]["inputs"]["datastore"]
    )
    images_folder = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_datastore_fullpath,
        description="Data to be blurred",
    )

    model_string = (
        "azureml:"
        f"{settings['inference_pipeline']['inputs']['model_name']}:"
        f"{settings['inference_pipeline']['inputs']['model_version']}"
    )
    model_input = Input(
        type=AssetTypes.CUSTOM_MODEL,
        path=model_string,
        description="Model weights for evaluation",
    )

    detect_and_blur_sensitive_data_step = detect_and_blur_sensitive_data(
        images_folder=images_folder,
        model=model_input,
    )

    batch_files_folder = os.path.join(
        input_datastore_fullpath,
        settings["inference_pipeline"]["inputs"]["inference_queue_rel_path"],
    )
    detect_and_blur_sensitive_data_step.outputs.batch_files_folder = Output(
        type="uri_folder",
        mode="rw_mount",
        path=batch_files_folder,
    )

    output_datastore_fullpath = aml_interface.get_datastore_full_path(
        settings["inference_pipeline"]["outputs"]["datastore"]
    )
    detect_and_blur_sensitive_data_step.outputs.output_folder = Output(
        type="uri_folder", mode="rw_mount", path=output_datastore_fullpath
    )
    return {}


aml_interface = AMLInterface()


def main(n_jobs: int = 1):
    for i in range(n_jobs):
        print(f"\n==> Starting job {i + 1}/{n_jobs}...")
        aml_interface.submit_pipeline_experiment(
            pipeline_function=inference_pipeline,
            experiment_name=settings["aml_experiment_details"]["experiment_name"],
            default_compute=settings["aml_experiment_details"]["compute_name"],
            show_log=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_jobs", default="1", help="Number of identical jobs to start."
    )
    args = parser.parse_args()
    n_jobs = int(args.n_jobs)
    main(n_jobs=n_jobs)
