import os
import sys
import random
import re
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import json

from sqlalchemy.exc import SQLAlchemyError

sys.path.append("../../..")

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)

from yolov5.baas_utils.database_handler import DBConfigSQLAlchemy  # noqa: E402
from yolov5.baas_utils.database_tables import DetectionInformation  # noqa: E402

logger = logging.getLogger(__name__)

class SmartSampling:
    def __init__(self, input_structured_folder, customer_quality_check_folder, customer_retraining_folder, 
                 database_parameters_json, customer_name, sampling_parameters):
        self.input_structured_folder = input_structured_folder
        self.customer_quality_check_folder = customer_quality_check_folder
        self.customer_retraining_folder = customer_retraining_folder
        self.database_parameters_json = database_parameters_json
        self.customer_name = customer_name
        self.sampling_parameters = sampling_parameters
        
    def sample_images_for_quality_check(self, grouped_images_by_date: Dict[str, List[str]],
                                        input_structured_folder: str, customer_quality_check_folder: str, 
    ) -> None:
        """
        Samples a specified number of images from each date for quality checking.

        Parameters
        ----------
        grouped_images_by_date : Dict[str, List[str]]
            A dictionary where keys are dates and values are lists of image file names from those dates.
        input_structured_folder : str
            The path of the input folder containing images.
        customer_quality_check_folder : str
            The destination folder path for the quality check images.
        n_images_to_sample : int
            Number of images to sample for each date.
        """
        
        logger.info(f'Sampling {n_images_to_sample} images for quality check.. \n')
        
        # Sample a number of random images for manual quality check
        # The number is set in config.yml as quality_check_sample_size
        n_images_to_sample = self.sampling_parameters["quality_check_sample_size"]
        
        quality_check_images = get_n_random_images_per_date(grouped_images_by_date, n_images_to_sample)
        
        logger.info(f'Quality check images: {quality_check_images} \n')

        for key, values in quality_check_images.items():
            for value in values:
                copy_file(
                    "/" + key + "/" + value, str(input_structured_folder), str(customer_quality_check_folder)
                )
        

    def collect_images_above_threshold_from_db(self, database_parameters_json: str, grouped_images_by_date: Dict[str, List[str]], 
                                               customer_name: str
    ) -> Tuple[Dict[datetime, List[Dict]], Dict[Tuple[str, datetime, str], int]]:
        """
        Collects images with detections above a specified confidence score threshold from the database.

        Parameters
        ----------
        database_parameters_json : str
            JSON string with database connection parameters.
        grouped_images_by_date : Dict[str, List[str]]
            A dictionary mapping dates to lists of image filenames.
        customer_name : str
            The name of the customer for whom the images belong.
        conf_score_threshold : float
            The threshold for the confidence score above which detections are considered.

        Returns
        -------
        Tuple[Dict[datetime, List[Dict]], Dict[Tuple[str, datetime, str], int]]
            A tuple containing two elements:
            - A dictionary with dates as keys and lists of dictionaries containing detection information as values.
            - A dictionary mapping tuples of customer name, upload date, and image name to their detection counts.

        Raises
        ------
        SQLAlchemyError
            If there is a failure in database operations, such as a connection issue or a query execution error.
        ValueError
            If there is an error in processing or parsing the data.
        Exception
            For any other unexpected errors that might occur during the execution of the function.
        """
        
        images_statistics = {}
        image_counts = {}

        try:
            
            # Connect to the database
            database_parameters = json.loads(database_parameters_json)
            db_username = database_parameters["db_username"]
            db_name = database_parameters["db_name"]
            db_hostname = database_parameters["db_hostname"]
            
            # Validate if database credentials are provided
            if not db_username or not db_name or not db_hostname:
                raise ValueError("Please provide database credentials.")
            
            # Create a DBConfigSQLAlchemy object
            db_config = DBConfigSQLAlchemy(db_username, db_hostname, db_name)
            
            # Create the database connection
            db_config.create_connection()
        
            with db_config.managed_session() as session:
                for upload_date, image_names in grouped_images_by_date.items():
                    logger.info(f'Upload Date: {upload_date} \n')
                    upload_date = datetime.strptime(upload_date, "%Y-%m-%d_%H_%M_%S")
                    logger.info(f'Formatted Upload Date: {upload_date} \n')
                    for image_name in image_names:
                        query = session.query(DetectionInformation).filter(
                            DetectionInformation.image_customer_name == customer_name,
                            DetectionInformation.image_upload_date == upload_date,
                            DetectionInformation.image_filename == image_name,
                            DetectionInformation.conf_score > self.conf_score_threshold
                        )
                        results = query.all()
                        count = len(results)

                        # Populating image_counts
                        image_key = (customer_name, upload_date, image_name)
                        image_counts[image_key] = count

                        # Populating images_statistics with detailed information
                        if results:
                            extracted_data = [result.__dict__ for result in results]
                            if upload_date in images_statistics:
                                images_statistics[upload_date].extend(extracted_data)
                            else:
                                images_statistics[upload_date] = extracted_data
                                
        except SQLAlchemyError as e:
            logger.error(f"Database operation failed: {e}")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

        return images_statistics, image_counts