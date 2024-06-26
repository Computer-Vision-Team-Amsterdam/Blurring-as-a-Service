import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

sys.path.append("../../..")

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)

from cvtoolkit.database.baas_tables import DetectionInformation  # noqa: E402
from cvtoolkit.database.database_handler import DBConfigSQLAlchemy  # noqa: E402
from cvtoolkit.helpers.file_helpers import copy_file  # noqa: E402

logger = logging.getLogger(__name__)


class SmartSampler:
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

        Returns
        -------
        None
        """

        n_images_to_sample = self.sampling_parameters["quality_check_sample_size"]

        logger.info(f"Sampling {n_images_to_sample} images for quality check.. \n")

        quality_check_images = SmartSampler._get_n_random_images_per_date(
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
        self, date: str, grouped_images: List[str]
    ) -> None:
        """
        Samples images for retraining purposes from the structured input folder. The function first collects images
        from the database that are above a specified confidence score threshold. It then categorizes these images into
        bins based on detection counts and samples a specific ratio of images from each bin for every date. The sampled
        images are then copied to the customer retraining folder.

        Parameters
        ----------
        date: str
            Date (in 'YYYY-MM-DD_HH_MM_SS' string format)
        grouped_images : Dict[str, List[str]]
            Lists of image filenames.
            These filenames are used to query the database and collect detection data for the corresponding images.

        Returns
        -------
        None
        """

        df_images = self._collect_images_above_threshold_from_db(date, grouped_images)
        logger.info(df_images.head())

        if df_images.empty:
            logger.warning("The dataframe containing the images is empty.")
            return None

        df_images, bin_counts = SmartSampler._categorize_images_into_bins(df_images)

        for bin_label, images in bin_counts.items():
            logger.info(
                f"Number of images with detections in bin {bin_label}: {len(images)}"
            )

        ratio = self.sampling_parameters["sampling_ratio"]
        percentage_ratio = ratio / 100
        sampled_images_df = SmartSampler._sample_images_equally_from_bins(
            df_images, percentage_ratio
        )

        for _, row in sampled_images_df.iterrows():
            formatted_upload_date = row["image_upload_date"]
            image_filename = row["image_filename"]
            copy_file(
                f"/{formatted_upload_date}/{image_filename}",
                str(self.input_structured_folder),
                str(self.customer_retraining_folder),
            )
        logger.info(f"Sampled for retraining: {sampled_images_df.count()} images")

    def _collect_images_above_threshold_from_db(
        self, date: str, grouped_images: List[str]
    ) -> pd.DataFrame:
        """
        Collects images with detections above a specified confidence score threshold from the database.

        Parameters
        ----------
        date: str
            Date (in 'YYYY-MM-DD_HH_MM_SS' string format)
        grouped_images : Dict[str, List[str]]
            Lists of image filenames.

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

        try:
            # Connect to the database
            database_parameters = json.loads(self.database_parameters_json)
            db_username = database_parameters["db_username"]
            db_name = database_parameters["db_name"]
            db_hostname = database_parameters["db_hostname"]
            client_id = database_parameters["client_id"]

            if not db_username or not db_name or not db_hostname:
                raise ValueError("Please provide database credentials.")

            db_config = DBConfigSQLAlchemy(db_username, db_hostname, db_name, client_id)

            db_config.create_connection()

            conf_score_threshold = self.sampling_parameters["conf_score_threshold"]

            with db_config.managed_session() as session:
                logger.info(f"Upload Date: {date} \n")
                upload_date_datetime = datetime.strptime(date, "%Y-%m-%d_%H_%M_%S")

                query = (
                    session.query(
                        DetectionInformation.image_upload_date,
                        DetectionInformation.image_filename,
                        func.count(DetectionInformation.id).label("count"),
                    )
                    .filter(
                        DetectionInformation.image_customer_name == self.customer_name,
                        DetectionInformation.image_upload_date == upload_date_datetime,
                        DetectionInformation.image_filename.in_(grouped_images),
                        DetectionInformation.conf_score > conf_score_threshold,
                    )
                    .group_by(
                        DetectionInformation.image_customer_name,
                        DetectionInformation.image_upload_date,
                        DetectionInformation.image_filename,
                    )
                    .statement
                )
                results = pd.read_sql_query(sql=query, con=db_config.engine)

                logger.info(f"Count results DB: {results.shape[0]} \n")

                if not results.empty:
                    results["image_upload_date"] = pd.to_datetime(
                        results["image_upload_date"]
                    )
                    results["image_upload_date"] = results[
                        "image_upload_date"
                    ].dt.strftime("%Y-%m-%d_%H_%M_%S")
                return results

        except SQLAlchemyError as e:
            logger.error(f"Database operation failed: {e}")
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    @staticmethod
    def _get_n_random_images_per_date(
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
    def _categorize_images_into_bins(
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
        logger.info(f"Number of unique images: {len(df)}")

        min_count, max_count = df["count"].min(), df["count"].max()
        logger.info(f"Minimum number of detections for an image: {min_count}")
        logger.info(f"Maximum number of detections for an image: {max_count}")

        detection_range = max_count - min_count
        bin_size = SmartSampler._determine_bin_size(detection_range)

        # Calculate the bin edges
        bins = np.linspace(min_count, max_count, bin_size + 1)

        bin_counts = {}

        bin_labels = SmartSampler._create_bin_labels(bins)

        # Categorize images into bins
        df["bin_label"] = pd.cut(
            df["count"], bins, labels=bin_labels, include_lowest=True, right=True
        )

        for label in bin_labels:
            bin_counts[label] = df[df["bin_label"] == label]

        return df, bin_counts

    @staticmethod
    def _sample_images_equally_from_bins(
        df: pd.DataFrame, percentage_ratio: float
    ) -> pd.DataFrame:
        """
        Samples a percentage of images equally from each bin for each date.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing image data with columns including
            'image_upload_date', 'image_filename', 'bin_label'.
        percentage_ratio : float
            The ratio of total images to sample from each date.

        Returns
        -------
        pd.DataFrame
            A DataFrame of sampled images.
        """

        sampled_images_df = pd.DataFrame()

        unique_dates = df["image_upload_date"].unique()
        logger.info(f"Unique dates: {unique_dates}")

        for upload_date in unique_dates:
            df_date = df[df["image_upload_date"] == upload_date]
            total_images = len(df_date)
            logger.info(f"Total images on date {upload_date}: {total_images}")

            total_images_to_sample = int(total_images * percentage_ratio)

            # Ensure at least one image is sampled if total_images_to_sample > 0
            total_images_to_sample = (
                max(total_images_to_sample, 1) if total_images > 0 else 0
            )

            logger.info(
                f"Total images to sample on date {upload_date}: {total_images_to_sample}"
            )

            if "bin_label" not in df_date.columns:
                logger.error(
                    "bin_label column not found in DataFrame for date: "
                    + str(upload_date)
                )
            else:
                bin_labels = df_date["bin_label"].unique()
                logger.debug(f"Bin labels for date {upload_date}: {bin_labels}")

                images_per_bin = total_images_to_sample // len(bin_labels)
                remainder = total_images_to_sample % len(bin_labels)
                logger.info(f"Images per bin: {images_per_bin}, Remainder: {remainder}")

                sampled_images_df = SmartSampler._calculate_n_of_images_to_sample(
                    sampled_images_df, df_date, bin_labels, images_per_bin, remainder
                )

                logger.debug(
                    f"Sampled images for date {upload_date}: {sampled_images_df.head()}"
                )

        return sampled_images_df

    @staticmethod
    def _determine_bin_size(detection_range: int) -> int:
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

        if detection_range > 50:
            return 10
        if detection_range > 10:
            return 5
        return 3

    @staticmethod
    def _create_bin_labels(bins):
        """
        Creates bin labels based on the given bins array.

        Parameters
        ----------
        bins : np.ndarray
            An array containing the bin edges.

        Examples
        --------
        >>> bins = [0, 5, 10]
        >>> SmartSampler._create_bin_labels(bins)
        ['0-4', '5-9']

        Returns
        -------
        List[str]
            A list of bin labels, where each label represents a range in the format "start-end".
        """
        bin_labels = [
            f"{int(bins[i])}-{int(bins[i + 1]) - 1}" for i in range(len(bins) - 1)
        ]
        return bin_labels

    @staticmethod
    def _calculate_n_of_images_to_sample(
        sampled_images_df, df_date, bin_labels, images_per_bin, remainder
    ):
        """
        Calculates the number of images to sample from each bin, considering the base number of images per bin and the remainder.

        This method iterates over each bin and determines the number of images to sample from that bin.
        The calculation is based on a base number of images per bin (`images_per_bin`) and a remainder
        that accounts for any extra images to be distributed evenly across the bins.

        Parameters
        ----------
        sampled_images_df : pd.DataFrame
            An empty DataFrame to which sampled images are appended.
        df_date : pd.DataFrame
            A DataFrame filtered for a specific date, containing image data including 'bin_label'.
        bin_labels : list
            A list of unique bin labels indicating different categories or bins.
        images_per_bin : int
            The base number of images to sample from each bin, calculated as the total
            number of images to be sampled divided by the number of bins.
        remainder : int
            The number of extra images that need to be distributed evenly across the bins.

        Returns
        -------
        pd.DataFrame
            A DataFrame of sampled images, concatenated from each bin.

        Notes
        -----
        - For each bin, the method first calculates `num_to_sample` which is the minimum of `images_per_bin + 1`
        (if there is a remainder) and the total number of images in that bin.
        - If a remainder exists, it is decreased by 1 after allocating an extra image to a bin,
        ensuring even distribution of the extra images across bins.
        - The method samples `num_to_sample` images from each bin and appends them to the resulting DataFrame.

        Example
        -------
        # Assuming a DataFrame 'df_date' for a specific date with 'bin_label' column
        # and an array 'bin_labels' containing unique bin labels, e.g., ['[0-9]', '[10-19]', '[20-29]']
        df_date = pd.DataFrame({
            'image_filename': ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg'],
            'bin_label': ['[0-9]', '[0-9]', '[10-19]', '[10-19]', '[20-29]']
        })
        bin_labels = ['[0-9]', '[10-19]', '[20-29]']
        images_per_bin = 1  # base number of images per bin
        remainder = 2  # extra images to distribute

        sampled_images_df = SmartSampler._calculate_n_of_images_to_sample(df_date, bin_labels, images_per_bin, remainder)
        print(sampled_images_df)
        """
        for bin_label in bin_labels:
            df_bin = df_date[df_date["bin_label"] == bin_label]

            # Adjust the number of images to sample from this bin
            num_to_sample = min(
                images_per_bin + (1 if remainder > 0 else 0), len(df_bin)
            )
            remainder -= 1 if remainder > 0 else 0

            sampled_from_bin = df_bin.sample(n=num_to_sample)
            sampled_images_df = sampled_images_df.append(sampled_from_bin)

        return sampled_images_df
