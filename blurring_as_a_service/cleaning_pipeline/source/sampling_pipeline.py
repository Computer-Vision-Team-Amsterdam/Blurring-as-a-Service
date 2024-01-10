import os
import sys
import random
import re
import numpy as np
import logging
import pandas as pd
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

from blurring_as_a_service.utils.generics import (  # noqa: E402
    copy_file,
)

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
        
        # Sample a number of random images for manual quality check
        # The number is set in config.yml as quality_check_sample_size
        n_images_to_sample = self.sampling_parameters["quality_check_sample_size"]
        
        logger.info(f'Sampling {n_images_to_sample} images for quality check.. \n')
        
        quality_check_images = SmartSampling.get_n_random_images_per_date(grouped_images_by_date, n_images_to_sample)
        
        logger.info(f'Quality check images: {quality_check_images} \n')

        for key, values in quality_check_images.items():
            for value in values:
                copy_file(
                    "/" + key + "/" + value, str(input_structured_folder), str(customer_quality_check_folder)
                )
        
    def sample_images_for_retraining(self, grouped_images_by_date: Dict[str, List[str]]) -> None:
        
        # Collect images above the confidence score threshold from the database
        _, image_counts = SmartSampling.collect_images_above_threshold_from_db(self.database_parameters_json, grouped_images_by_date, self.customer_name)
        
        # Group images into bins
        bin_counts, _ = SmartSampling.categorize_images_into_bins(image_counts)

        # Count images in each bin
        for bin_label, images in bin_counts.items():
            logger.info(f"Number of images with detections in bin {bin_label}: {len(images)}")
            
        # Sample a ratio of the images for each date
        # The ratio is set in config.yml as sampling_ratio
        ratio = self.sampling_parameters["sampling_ratio"]
        percentage_ratio = ratio / 100
        
        sampled_images_by_date = SmartSampling.sample_images_equally_from_bins(
            image_counts, bin_counts, percentage_ratio
        )
        
        logger.info(f'Sampled images by date: {sampled_images_by_date} \n')
        
        for upload_date, images in sampled_images_by_date.items():
            # Copy the sampled images
            for image in images:
                formatted_upload_date = upload_date.strftime("%Y-%m-%d_%H_%M_%S")
                image_filename = image[2]  # Assuming image is a tuple (customer_name, upload_date, image_name)
                copy_file(
                    f"/{formatted_upload_date}/{image_filename}", str(self.input_structured_folder), str(self.customer_retraining_folder)
                )
                # Optionally, print out the image names being sampled for debugging
                logger.info(f"Sampled for retraining: /{formatted_upload_date}/{image_filename}")
    
    @staticmethod
    def collect_images_above_threshold_from_db(self, grouped_images_by_date: Dict[str, List[str]]
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
            database_parameters = json.loads(self.database_parameters_json)
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
                            DetectionInformation.image_customer_name == self.customer_name,
                            DetectionInformation.image_upload_date == upload_date,
                            DetectionInformation.image_filename == image_name,
                            DetectionInformation.conf_score > self.conf_score_threshold
                        )
                        results = query.all()
                        count = len(results)

                        # Populating image_counts
                        image_key = (self.customer_name, upload_date, image_name)
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
    
    @staticmethod
    def get_n_random_images_per_date(grouped_images_by_date: Dict[str, List[str]], n_images_to_sample: int) -> Dict[str, List[str]]:
        """
        Randomly samples a specified number of images for each date.

        Parameters
        ----------
        grouped_images_by_date : Dict[str, List[str]]
            A dictionary where keys are dates and values are lists of image file names from those dates.
        n_images_to_sample : int
            The number of images to randomly sample from each date's list.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary where keys are dates and values are the randomly sampled image file names.
        """
        
        random_result = {}

        for key, values in grouped_images_by_date.items():
            if len(values) >= n_images_to_sample:
                random_values = random.sample(values, n_images_to_sample)
            else:
                random_values = values
            random_result[key] = random_values

        return random_result

    @staticmethod
    def categorize_images_into_bins(image_counts: Dict[str, int]) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Categorizes images into bins based on their detection counts.

        Parameters
        ----------
        image_counts : Dict[str, int]
            A dictionary mapping image identifiers to their respective detection counts.

        Returns
        -------
        Tuple[Dict[str, List[str]], List[str]]
            A tuple containing two elements:
            - A dictionary where keys are bin labels and values are lists of image identifiers in that bin.
            - A list of bin labels.

        If no detections are found, an empty dictionary and list are returned.
        """
        
        if not image_counts:
            logger.info("No detections found for the given criteria.")
            return {}, []

        # Calculate min and max counts
        counts = image_counts.values()
        min_count, max_count = min(counts), max(counts)
        logger.info(f"Minimum number of detections for an image: {min_count}")
        logger.info(f"Maximum number of detections for an image: {max_count}")

        # Determine the range and define bin size strategy
        detection_range = max_count - min_count
        bin_size = SmartSampling.determine_bin_size(detection_range)

        # Calculate the bin edges
        bins = np.linspace(min_count, max_count, bin_size + 1)

        # Initialize a dictionary to hold bin counts
        bin_counts, bin_labels = SmartSampling.initialize_bin_counts(bins)

        # Categorize images into bins
        SmartSampling.categorize_into_bins(image_counts, bins, bin_labels, bin_counts)

        return bin_counts, bin_labels

    @staticmethod
    def sample_images_equally_from_bins(
        image_counts: Dict[Tuple[str, datetime, str], int], 
        bin_counts: Dict[str, List[Tuple[str, datetime, str]]], 
        percentage_ratio: float
    ) -> Dict[datetime, List[Tuple[str, datetime, str]]]:
        """
        Samples a percentage of images equally from each bin for each date.

        Parameters
        ----------
        image_counts : Dict[Tuple[str, datetime, str], int]
            A dictionary mapping image tuples to their detection counts. Each tuple contains customer name, 
            upload date, and image name.
        bin_counts : Dict[str, List[Tuple[str, datetime, str]]]
            A dictionary where keys are bin labels and values are lists of image tuples in that bin.
        percentage_ratio : float
            The ratio of total images to sample from each date.

        Returns
        -------
        Dict[datetime, List[Tuple[str, datetime, str]]]
            A dictionary mapping dates to lists of sampled image tuples.
        """
        
        sampled_images_by_date = {}

        # Extract all unique upload dates from image_counts
        unique_dates = {img[1] for img in image_counts.keys()}  # Extracting the upload_date part of the tuple
        logger.info(f'Unique dates: {unique_dates}')

        for upload_date in unique_dates:
            # Filter image_counts for the current date and count unique images
            unique_images_on_date = {k for k in image_counts.keys() if k[1] == upload_date}
            total_images = len(unique_images_on_date)
            logger.info(f'Total images on date {upload_date}: {total_images}')
            total_images_to_sample = int(total_images * percentage_ratio)
            
            # Ensure at least one image is sampled if total_images_to_sample > 0
            total_images_to_sample = max(total_images_to_sample, 1) if total_images > 0 else 0
            
            logger.info(f'Total images to sample on date {upload_date}: {total_images_to_sample}')

            # Calculate the number of images to sample per bin
            images_per_bin = total_images_to_sample // len(bin_counts)
            logger.info(f'Images per bin: {images_per_bin}')
            remainder = total_images_to_sample % len(bin_counts)
            logger.info(f'Remainder: {remainder}')

            sampled_images = []

            for bin_label, images_in_bin in bin_counts.items():
                # Extract the unique images for the current bin and date
                unique_images_in_bin = [img for img in images_in_bin if img[1] == upload_date]
                
                # Adjust the number of images to sample from this bin
                if images_per_bin > 0 or (remainder > 0 and unique_images_in_bin):
                    num_to_sample = min(images_per_bin + (1 if remainder > 0 else 0), len(unique_images_in_bin))
                    remainder -= 1 if remainder > 0 else 0
                else:
                    # If images_per_bin is 0 and no remainder, skip this bin
                    continue 

                # Sample images
                sampled_from_bin = random.sample(unique_images_in_bin, num_to_sample) if unique_images_in_bin else []
                sampled_images.extend(sampled_from_bin)

            sampled_images_by_date[upload_date] = sampled_images

        return sampled_images_by_date
    
    @staticmethod
    def determine_bin_size(detection_range: int) -> int:
        """
        Determines the bin size for categorization based on the detection range.

        Parameters
        ----------
        detection_range : int
            The range of detection counts across all images.

        Returns
        -------
        int
            The number of bins to be used for categorization.
        """
        
        if detection_range <= 10:
            return 3
        elif 10 < detection_range <= 50:
            return 5
        else:
            return 10

    @staticmethod
    def initialize_bin_counts(bins: np.ndarray) -> Tuple[Dict[str, List], List[str]]:
        """
        Initializes a dictionary (bin_counts) and a list (bin_labels) that will be used 
        for categorizing images into bins based on their detection counts. 

        Parameters
        ----------
        bins : np.ndarray
            The array of bin edges.

        Returns
        -------
        Tuple[Dict[str, List], List[str]]
            A tuple containing two elements:
            - A dictionary where keys are bin labels and values are empty lists for each bin.
            - A list of bin labels.
        """
        
        bin_counts = {}
        
        # Iterate through the given bins array and create a list of bin labels. 
        # Each label represents a range, formatted as "start-end", 
        # where "start" is the beginning of a bin and "end" is one less than the start of the next bin.
        bin_labels = [f"{int(bins[i])}-{int(bins[i + 1]) - 1}" for i in range(len(bins) - 1)]
        
        # Create a dictionary (bin_counts) with keys being the bin labels and values being empty lists. 
        # Each list will contain image identifiers that fall within the corresponding bin's range. 
        for label in bin_labels:
            bin_counts[label] = []
            
        return bin_counts, bin_labels

    @staticmethod
    def categorize_into_bins(image_counts: Dict[str, int], bins: np.ndarray, bin_labels: List[str], bin_counts: Dict[str, List[str]]) -> None:
        """
        Categorizes each image into a bin based on its detection count.

        Parameters
        ----------
        image_counts : Dict[str, int]
            A dictionary mapping image identifiers to their detection counts.
        bins : np.ndarray
            The array of bin edges.
        bin_labels : List[str]
            A list of bin labels.
        bin_counts : Dict[str, List[str]]
            A dictionary to hold the categorized images, with keys as bin labels and values as lists of images.

        Returns
        -------
        None
            This function modifies the bin_counts dictionary in place.
        """
        
        for image, count in image_counts.items():
            bin_index = np.digitize(count, bins, right=True) - 1
            bin_label = bin_labels[bin_index]
            bin_counts[bin_label].append(image)