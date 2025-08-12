import logging
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from typing import List, Set

from azure.ai.ml.constants import AssetTypes
from azureml.core import Run
from mldesigner import Input, Output, command_component
from sqlalchemy.exc import SQLAlchemyError

sys.path.append("../../..")

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

settings = BlurringAsAServiceSettings.set_from_yaml(config_path)

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from cvtoolkit.database.baas_tables import (  # noqa: E402
    BatchRunInformation,
    ImageProcessingStatus,
)
from cvtoolkit.database.database_handler import DBConfigSQLAlchemy  # noqa: E402
from cvtoolkit.helpers.file_helpers import delete_file  # noqa: E402
from cvtoolkit.multiprocessing.lock_file import LockFile  # noqa: E402

from blurring_as_a_service.inference_pipeline.source.baas_inference import (  # noqa: E402
    BaaSInference,
)
from blurring_as_a_service.inference_pipeline.source.db_utils import (  # noqa: E402
    create_db_connector,
)

aml_experiment_settings = settings["aml_experiment_details"]
run_id = Run.get_context().id


@command_component(
    name="detect_and_blur_sensitive_data",
    display_name="Detect and blur sensitive data from images",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def detect_and_blur_sensitive_data(
    images_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model: Input(type=AssetTypes.URI_FILE),  # type: ignore # noqa: F821
    batches_files_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to detect the areas to blur and blur those areas.

    Parameters
    ----------
    images_folder:
        Path of the mounted folder containing the images.
    model:
        Model weights for inference
    batches_files_path:
         Path to folder with multiple text files.
         One text file contains multiple rows.
         Each row is a relative path to {customer_name}_input_structured/inference_queue
    output_folder:
        Where to store the results.
    customer_name
        The name of the customer, with spaces replaced by underscores
    model_parameters_json
        All parameters used to run YOLOv5 inference in json format
    database_parameters_json
        Database credentials
    """
    start_time = get_current_time()
    logger = logging.getLogger("detect_and_blur_sensitive_data")
    if not os.path.exists(batches_files_path):
        raise FileNotFoundError(f"The folder '{batches_files_path}' does not exist.")
    output_rel_path = settings["inference_pipeline"]["outputs"]["output_rel_path"]
    if output_rel_path:
        output_folder = os.path.join(output_folder, output_rel_path)
    batch_files_to_iterate = os.listdir(batches_files_path)
    logging.info(f"Batches file to do: {batch_files_to_iterate}")
    error_trace = ""
    db_connector = create_db_connector()
    db_connector.create_connection()
    for batch_file_txt in batch_files_to_iterate:
        if batch_file_txt.endswith(".txt"):
            file_path = os.path.join(batches_files_path, batch_file_txt)
            try:
                if os.path.isfile(file_path):
                    logger.info(f"Creating inference step: {file_path}")
                    with LockFile(file_path) as src:
                        preprocessing_date = datetime.strptime(
                            re.search(
                                r"\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}", batch_file_txt
                            ).group(),
                            "%Y-%m-%d_%H_%M_%S",
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        folders_and_frames = create_dict_folders_and_frames_to_blur(
                            images_folder, src, preprocessing_date, db_connector
                        )
                        inference_pipeline = BaaSInference(
                            images_folder=images_folder,
                            output_folder=output_folder,
                            model_path=model,
                            inference_settings=settings["inference_pipeline"],
                            folders_and_frames=folders_and_frames,
                            customer_name=settings["customer"],
                            image_upload_date=preprocessing_date,
                        )

                        inference_pipeline.run_pipeline()
                    delete_file(file_path)
            except FileNotFoundError as e:
                logger.warning(
                    f"File {file_path} not found: {e}, if running in parallel, this could be expected."
                )
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                error_trace += f"{e}\n"

                try:
                    with db_connector.managed_session() as session:
                        batch_info = BatchRunInformation(
                            run_id=run_id,
                            start_time=start_time,
                            end_time=get_current_time(),
                            trained_yolo_model=os.path.split(model)[-1],
                            success=False,
                            error_code=e,
                        )
                        session.add(batch_info)
                except SQLAlchemyError as e:
                    db_connector.close_connection()
                    raise e

    try:
        with db_connector.managed_session() as session:
            batch_info = BatchRunInformation(
                run_id=run_id,
                start_time=start_time,
                end_time=get_current_time(),
                trained_yolo_model=os.path.split(model)[-1],
                success=not bool(error_trace),
                error_code=error_trace if error_trace else None,
            )
            session.add(batch_info)
    except SQLAlchemyError as e:
        db_connector.close_connection()
        raise e
    db_connector.close_connection()


def create_dict_folders_and_frames_to_blur(
    input_structured_folder: str,
    src: List[str],
    preprocessing_date: str,
    db_connector: DBConfigSQLAlchemy,
) -> defaultdict[str, List[str]]:
    """
    Create a dictionary mapping folders to frames that need to be blurred.

    This function reads lines from the source list, filters out already processed images,
    and constructs a dictionary where the keys are folder paths and the values are lists
    of frames that need to be blurred.

    Parameters
    ----------
    input_structured_folder : str
        Path of the folder containing the images.
    src : List[str]
        List of image paths to be processed.
    preprocessing_date : str
        The date for which to fetch the processed images.
    db_connector : DBConfigSQLAlchemy
        A configuration object for connecting to the database.

    Returns
    -------
    defaultdict
        A dictionary where keys are folder paths and values are lists of frames to be blurred.
    """
    processed_images = fetch_already_processed_images(preprocessing_date, db_connector)
    folders_and_frames = defaultdict(list)
    for line in src:
        if line not in processed_images:
            parent_folder = line.split("/")[0]
            relative_path = line.split("/", 1)[1]
            folders_and_frames[f"{input_structured_folder}/{parent_folder}"].append(
                relative_path
            )
            lock_images_that_will_be_blurred(
                image_filename=line,
                image_upload_date=preprocessing_date,
                db_connector=db_connector,
            )
    return folders_and_frames


def fetch_already_processed_images(
    preprocessing_date: str, db_connector: DBConfigSQLAlchemy
) -> Set[str]:
    """
    Fetches the filenames of images that have already been processed on a given date.

    This function connects to a database using credentials specified in the settings,
    queries the database for images that have been processed or are in progress for a
    specific customer on the given preprocessing date, and returns a list of filenames
    of these images.

    Parameters
    ----------
    preprocessing_date : str
        The date for which to fetch the processed images.
    db_connector : DBConfigSQLAlchemy
        A configuration object for connecting to the database.

    Returns
    -------
    List[str]
        A list of filenames of images that have already been processed or are in progress.

    Raises
    ------
    ValueError
        If any of the required database credentials are missing or incomplete.
    SQLAlchemyError
        If there is an error querying the database.
    """
    processed_images = []
    with db_connector.managed_session() as session:
        try:
            processed_images = (
                session.query(
                    ImageProcessingStatus.image_upload_date,
                    ImageProcessingStatus.image_filename,
                )
                .filter(
                    ImageProcessingStatus.image_customer_name == settings["customer"],
                    ImageProcessingStatus.processing_status.in_(
                        ["inprogress", "processed"]
                    ),
                    ImageProcessingStatus.image_upload_date == preprocessing_date,
                )
                .all()
            )
        except SQLAlchemyError as e:
            raise e

    return {image.image_filename for image in processed_images}


def lock_images_that_will_be_blurred(
    image_filename: str,
    image_upload_date: str,
    db_connector: DBConfigSQLAlchemy,
) -> None:
    """
    Locks the images that will be blurred by updating the processing status in the database.

    Parameters
    ----------
    image_filename : str
        The filename of the image to be processed.
    image_upload_date : str
        The upload date of the image.

    Returns
    -------
    None
    """
    with db_connector.managed_session() as session:
        image_processing_status = ImageProcessingStatus(
            image_filename=image_filename,
            image_upload_date=image_upload_date,
            image_customer_name=settings["customer"],
            processing_status="inprogress",
        )
        session.add(image_processing_status)


def get_current_time():
    """
    Get the current time formatted as a string.

    Returns
    -------
        str: The current time in the format "YYYY-MM-DD HH:MM:SS".
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
