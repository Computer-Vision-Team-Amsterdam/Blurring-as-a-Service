import os

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.parallel import RunFunction, parallel_run_function

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


# Declare pipeline structure.
@pipeline(
    display_name="parallel job for batch inferencing",
)
def inference_pipeline():
    customer_name = settings["customer"]
    # model_name = inference_settings["model_name"]
    # model_version = inference_settings["model_version"]

    # Format the root path of the Blob Storage Container in Azure using placeholders
    input_structured_path = aml_interface.get_datastore_full_path(
        f"{customer_name}_input_structured"
    )

    # model_input = Input(
    #     type=AssetTypes.CUSTOM_MODEL,
    #     path=f"azureml:{model_name}:{model_version}",
    #     description="Model weights for evaluation",
    # )

    # Get the txt file that contains all paths of the files to run inference on
    batches_files_path = os.path.join(
        input_structured_path,
        "inference_queue",
    )

    input_with_txts = Input(
        type=AssetTypes.URI_FOLDER,
        path=batches_files_path,
        description="Data to be blurred",
    )

    # Declare parallel inferencing job.
    predict_yolo = batch_inferencing_with_mini_batch_size(
        job_data_path=input_with_txts,
        # score_model=model_input,
    )

    azureml_outputs_formatted = aml_interface.get_datastore_full_path(
        f"{customer_name}_output"
    )

    # User could override parallel job run-level property when invoke that parallel job/component in pipeline.
    predict_yolo.resources.instance_count = 2
    predict_yolo.max_concurrency_per_instance = 2
    predict_yolo.mini_batch_error_threshold = 10
    predict_yolo.outputs.job_output_file.path = os.path.join(
        azureml_outputs_formatted, "aggregated_returns.csv"
    )  # TODO I think we dont want this

    return {}


if __name__ == "__main__":
    # Retrieve values from the YAML
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    settings = BlurringAsAServiceSettings.get_settings()
    inference_settings = settings["inference_pipeline"]

    # Declare parallel job with run_function task
    # Based on https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/parallel/3a_mnist_batch_identification/mnist_batch_prediction.ipynb
    batch_inferencing_with_mini_batch_size = parallel_run_function(
        name="batch_inferencing_with_mini_batch_size",
        display_name="Batch Inferencing with mini_batch_size",
        description="parallel job to do batch inferencing with mini_batch_size on url folder with files input",
        tags={
            "azureml_parallel_example": "3a_sdk",
        },
        inputs=dict(
            job_data_path=Input(
                type=AssetTypes.URI_FOLDER,
                description="Input to txt files.",
                mode=InputOutputModes.RO_MOUNT,
            ),
            # score_model=Input(
            #     type=AssetTypes.CUSTOM_MODEL,
            #     description="The trained YOLOv5 model.",
            #     mode=InputOutputModes.DOWNLOAD,
            # ),
            outputs=dict(
                job_output_file=Output(
                    type=AssetTypes.URI_FILE,
                    mode=InputOutputModes.RW_MOUNT,
                ),
            ),
            input_data="${{inputs.job_data_path}}",  # Define which input data will be splitted into mini-batches
            mini_batch_size="5",
            # Use 'mini_batch_size' as the data division method. For files input data, this number define the file count for each mini-batch.
            instance_count=2,  # Use 2 nodes from compute cluster to run this parallel job.
            max_concurrency_per_instance=2,  # Create 2 worker processors in each compute node to execute mini-batches.
            error_threshold=5,
            # Monitor the failures of item processed by the gap between mini-batch input count and returns. 'Batch inferencing' scenario should return a list, dataframe, or tuple with the successful items to try to meet this threshold.
            mini_batch_error_threshold=5,
            # Monitor the failed mini-batch by exception, time out, or null return. When failed mini-batch count is higher than this setting, the parallel job will be marked as 'failed'.
            logging_level="DEBUG",
            environment_variables={
                "AZUREML_PARALLEL_EXAMPLE": "3a_sdk",
            },
            task=RunFunction(
                code="./source",
                entry_script="file_info.py",
                # environment=Environment(
                #     image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                #     conda_file="./script/environment_parallel.yml",
                # ), # TODO what to put here?
                # program_arguments="--model ${{inputs.score_model}} ",
                append_row_to="${{outputs.job_output_file}}",
                # Define where to output the aggregated returns from each mini-batches.
            ),
        ),
    )

    default_compute = settings["aml_experiment_details"][
        "compute_name"
    ]  # TODO use for example "cheap-cpu-cluster"
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        inference_pipeline, "inference_pipeline", default_compute
    )
