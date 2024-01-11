import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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

from blurring_as_a_service.utils.generics import copy_file  # noqa: E402

logger = logging.getLogger(__name__)


class SmartSampling:
    """
    A class designed to facilitate smart sampling of images for quality checks and retraining purposes in a
    machine learning pipeline. The class manages the sampling process by interfacing with a database to retrieve
    image detection data, categorizing images based on detection counts, and then sampling a subset of these images.

    Attributes
    ----------
    input_structured_folder : str
        The path to the input folder containing image data.
    customer_quality_check_folder : str
        The destination folder path where sampled images for quality checks are stored.
    customer_retraining_folder : str
        The destination folder path where sampled images for retraining are stored.
    database_parameters_json : str
        JSON string with database connection parameters.
    customer_name : str
        The name of the customer associated with the images.
    sampling_parameters : dict
        A dictionary containing parameters for the sampling process, such as sample size, confidence score
        threshold, and sampling ratio.

    Methods
    -------
    sample_images_for_quality_check(grouped_images_by_date: Dict[str, List[str]]) -> None
        Samples a specified number of images from each date for quality checks and stores them in the
        designated quality check folder.

    sample_images_for_retraining(grouped_images_by_date: Dict[str, List[str]]) -> None
        Samples images for retraining based on detection counts and confidence score thresholds, and stores
        them in the designated retraining folder.

    collect_images_above_threshold_from_db(grouped_images_by_date: Dict[str, List[str]]) -> pd.DataFrame
        Collects images with detections above a specified confidence score threshold from the database.

    get_n_random_images_per_date(grouped_images_by_date: Dict[str, List[str]], n_images_to_sample: int) -> Dict[str, List[str]]
        Randomly samples a specified number of images for each date.

    categorize_images_into_bins(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], List[str]]
        Categorizes images into bins based on their detection counts.

    sample_images_equally_from_bins(df: pd.DataFrame, percentage_ratio: float) -> pd.DataFrame
        Samples a percentage of images equally from each bin for each date.

    determine_bin_size(detection_range: int) -> int
        Determines the bin size for categorization based on the detection range.
    """

    def __init__(
        self,
        input_structured_folder,
        customer_quality_check_folder,
        customer_retraining_folder,
        database_parameters_json,
        customer_name,
        sampling_parameters,
    ):
        self.input_structured_folder = input_structured_folder
        self.customer_quality_check_folder = customer_quality_check_folder
        self.customer_retraining_folder = customer_retraining_folder
        self.database_parameters_json = database_parameters_json
        self.customer_name = customer_name
        self.sampling_parameters = sampling_parameters

    def sample_images_for_quality_check(
        self, grouped_images_by_date: Dict[str, List[str]]
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

        Returns
        -------
        None
        """

        # Sample a number of random images for manual quality check
        # The number is set in config.yml as quality_check_sample_size
        n_images_to_sample = self.sampling_parameters["quality_check_sample_size"]

        logger.info(f"Sampling {n_images_to_sample} images for quality check.. \n")

        quality_check_images = SmartSampling.get_n_random_images_per_date(
            grouped_images_by_date, n_images_to_sample
        )

        logger.info(f"Quality check images: {quality_check_images} \n")

        for key, values in quality_check_images.items():
            for value in values:
                copy_file(
                    "/" + key + "/" + value,
                    str(self.input_structured_folder),
                    str(self.customer_quality_check_folder),
                )

    def sample_images_for_retraining(
        self, grouped_images_by_date: Dict[str, List[str]]
    ) -> None:
        """
        Samples images for retraining purposes from the structured input folder. The function first collects images
        from the database that are above a specified confidence score threshold. It then categorizes these images into
        bins based on detection counts and samples a specific ratio of images from each bin for every date. The sampled
        images are then copied to the customer retraining folder.

        Parameters
        ----------
        grouped_images_by_date : Dict[str, List[str]]
            A dictionary mapping dates (in 'YYYY-MM-DD_HH_MM_SS' string format) to lists of image filenames.
            These filenames are used to query the database and collect detection data for the corresponding images.

        Returns
        -------
        None
        """

        # Collect images above the confidence score threshold from the database
        df_images = SmartSampling.collect_images_above_threshold_from_db(
            self, grouped_images_by_date
        )

        # Group images into bins
        df_images, bin_counts = SmartSampling.categorize_images_into_bins(df_images)

        # Count images in each bin
        for bin_label, images in bin_counts.items():
            logger.info(
                f"Number of images with detections in bin {bin_label}: {len(images)}"
            )

        # Sample a ratio of the images for each date
        # The ratio is set in config.yml as sampling_ratio
        ratio = self.sampling_parameters["sampling_ratio"]
        percentage_ratio = ratio / 100

        sampled_images_df = SmartSampling.sample_images_equally_from_bins(
            df_images, percentage_ratio
        )

        # Iterate over the sampled images DataFrame
        for _, row in sampled_images_df.iterrows():
            # Access image details from the row
            formatted_upload_date = row["image_upload_date"]
            image_filename = row["image_filename"]
            # Copy the sampled images
            copy_file(
                f"/{formatted_upload_date}/{image_filename}",
                str(self.input_structured_folder),
                str(self.customer_retraining_folder),
            )
            logger.info(
                f"Sampled for retraining: /{formatted_upload_date}/{image_filename}"
            )

    @staticmethod
    def collect_images_above_threshold_from_db(
        self, grouped_images_by_date: Dict[str, List[str]]
    ) -> pd.DataFrame:
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
        pd.DataFrame
            A DataFrame containing detailed information about each image detection,
            including image name, upload date, customer name, and count of detections.

        Raises
        ------
        SQLAlchemyError
            If there is a failure in database operations, such as a connection issue or a query execution error.
        ValueError
            If there is an error in processing or parsing the data.
        Exception
            For any other unexpected errors that might occur during the execution of the function.
        """

        images_statistics: Dict[str, list] = {}

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

            conf_score_threshold = self.sampling_parameters["conf_score_threshold"]

            with db_config.managed_session() as session:
                for upload_date, image_names in grouped_images_by_date.items():
                    logger.info(f"Upload Date: {upload_date} \n")
                    upload_date_datetime = datetime.strptime(
                        upload_date, "%Y-%m-%d_%H_%M_%S"
                    )
                    logger.info(f"Formatted Upload Date: {upload_date} \n")

                    for image_name in image_names:
                        query = session.query(DetectionInformation).filter(
                            DetectionInformation.image_customer_name
                            == self.customer_name,
                            DetectionInformation.image_upload_date
                            == upload_date_datetime,
                            DetectionInformation.image_filename == image_name,
                            DetectionInformation.conf_score > conf_score_threshold,
                        )
                        results = query.all()
                        count = len(results)

                        # Populating images_statistics with detailed information
                        if results:
                            extracted_data = [result.__dict__ for result in results]
                            for data in extracted_data:
                                data["count"] = count

                            if upload_date in images_statistics:
                                images_statistics[upload_date].extend(extracted_data)
                            else:
                                images_statistics[upload_date] = extracted_data

                # Flatten the dictionary to a list of dictionaries
                flat_list = [
                    item for sublist in images_statistics.values() for item in sublist
                ]

                # Convert the list of dictionaries to a DataFrame
                df = pd.DataFrame(flat_list)

                # Convert to datetime format
                df["image_upload_date"] = pd.to_datetime(df["image_upload_date"])

                # Format the date to the desired string format
                df["image_upload_date"] = df["image_upload_date"].dt.strftime(
                    "%Y-%m-%d_%H_%M_%S"
                )

        except SQLAlchemyError as e:
            logger.error(f"Database operation failed: {e}")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

        return df

    @staticmethod
    def get_n_random_images_per_date(
        grouped_images_by_date: Dict[str, List[str]], n_images_to_sample: int
    ) -> Dict[str, List[str]]:
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
    def categorize_images_into_bins(
        df: pd.DataFrame,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Categorizes images into bins based on their detection counts.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing image data with columns for detection counts.

        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            A tuple containing two elements:
            - A dictionary where keys are bin labels and values are DataFrames of image identifiers in that bin.
            - A list of bin labels.
        """

        if df.empty:
            logger.info("No detections found for the given criteria.")
            return {}, {}

        # Diagnostic log to check the initial DataFrame structure
        logger.debug(f"Initial DataFrame structure: {df.head()}")

        # Drop duplicates based on the unique triple and create a copy of the DataFrame
        unique_df = df.drop_duplicates(
            subset=["image_upload_date", "image_customer_name", "image_filename"]
        ).copy()

        # Log the number of unique images
        logger.info(f"Number of unique images: {len(unique_df)}")

        # Calculate min and max counts
        min_count, max_count = unique_df["count"].min(), unique_df["count"].max()
        logger.info(f"Minimum number of detections for an image: {min_count}")
        logger.info(f"Maximum number of detections for an image: {max_count}")

        # Determine the range and define bin size strategy
        detection_range = max_count - min_count
        bin_size = SmartSampling.determine_bin_size(detection_range)

        # Calculate the bin edges
        bins = np.linspace(min_count, max_count, bin_size + 1)

        # Initialize a dictionary to hold bin counts and labels
        bin_counts = {}

        # Iterate through the given bins array and create a list of bin labels.
        # Each label represents a range, formatted as "start-end",
        # where "start" is the beginning of a bin and "end" is one less than the start of the next bin.
        bin_labels = [
            f"{int(bins[i])}-{int(bins[i + 1]) - 1}" for i in range(len(bins) - 1)
        ]

        # Categorize images into bins
        unique_df["bin_label"] = pd.cut(
            unique_df["count"], bins, labels=bin_labels, include_lowest=True, right=True
        )

        for label in bin_labels:
            bin_counts[label] = unique_df[unique_df["bin_label"] == label]

        return unique_df, bin_counts

    @staticmethod
    def sample_images_equally_from_bins(
        df: pd.DataFrame, percentage_ratio: float
    ) -> pd.DataFrame:
        """
        Samples a percentage of images equally from each bin for each date.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing image data with columns including 'image_upload_date', 'image_customer_name', 'image_filename', 'bin_label'.
        percentage_ratio : float
            The ratio of total images to sample from each date.

        Returns
        -------
        pd.DataFrame
            A DataFrame of sampled images.
        """

        # Initialize the DataFrame to store sampled images
        sampled_images_df = pd.DataFrame()

        # Extract all unique upload dates from the DataFrame
        unique_dates = df["image_upload_date"].unique()
        logger.info(f"Unique dates: {unique_dates}")

        for upload_date in unique_dates:
            # Filter for the current date
            df_date = df[df["image_upload_date"] == upload_date]
            total_images = len(df_date)
            logger.info(f"Total images on date {upload_date}: {total_images}")

            # Calculate the total number of images to sample
            total_images_to_sample = int(total_images * percentage_ratio)

            # Ensure at least one image is sampled if total_images_to_sample > 0
            total_images_to_sample = (
                max(total_images_to_sample, 1) if total_images > 0 else 0
            )

            logger.info(
                f"Total images to sample on date {upload_date}: {total_images_to_sample}"
            )

            # Check if 'bin_label' column exists
            if "bin_label" not in df_date.columns:
                logger.error(
                    "bin_label column not found in DataFrame for date: "
                    + str(upload_date)
                )
                continue  # Skip to next date or handle error as needed

            # Get the unique bin labels
            bin_labels = df_date["bin_label"].unique()
            logger.debug(f"Bin labels for date {upload_date}: {bin_labels}")

            # Calculate the number of images to sample per bin
            images_per_bin = total_images_to_sample // len(bin_labels)
            remainder = total_images_to_sample % len(bin_labels)
            logger.info(f"Images per bin: {images_per_bin}, Remainder: {remainder}")

            for bin_label in bin_labels:
                df_bin = df_date[df_date["bin_label"] == bin_label]

                # Adjust the number of images to sample from this bin
                num_to_sample = min(
                    images_per_bin + (1 if remainder > 0 else 0), len(df_bin)
                )
                remainder -= 1 if remainder > 0 else 0

                # Sample images
                sampled_from_bin = df_bin.sample(n=num_to_sample)
                sampled_images_df = sampled_images_df.append(sampled_from_bin)

            # Print statement for debugging
            logger.debug(
                f"Sampled images for date {upload_date}: {sampled_images_df.head()}"
            )

        # Return the full DataFrame after processing all dates
        return sampled_images_df

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
