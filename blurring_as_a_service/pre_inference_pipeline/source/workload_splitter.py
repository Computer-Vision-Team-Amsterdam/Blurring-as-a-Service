import csv
import logging
import math
import os

logger = logging.getLogger(__name__)

from blurring_as_a_service.pre_inference_pipeline.source.image_paths import (  # noqa: E402
    get_image_paths,
)


class WorkloadSplitter:
    @staticmethod
    def create_batches(
        data_folder: str,
        datastore_input_path: str,
        number_of_batches: int,
        exclude_file: str,
        output_folder: str,
        execution_time: str,
    ) -> None:
        """
        Starting from a data folder, iterates over all subfolders and equally groups all jpg files into number_of_batches
        batches. These groups are stored into multiple txt files where each line is a file including the relative path.

        Examples
        --------
        data_folder > folder_1 > img1.jpg
                               > img2.jpg
                    > folder_2 > img3.jpg
        number_of_batches = 3
        output_folder/batch_0.txt
            folder_1/img1.jpg
        output_folder/batch_1.txt
            folder_1/img2.jpg
        output_folder/batch_2.txt
            folder_2/img3.jpg

        Parameters
        ----------
        data_folder : str
            Root folder containing the images.
        number_of_batches : int
            Number of files to distribute the data.
        exclude_file : Optional[str]
            CSV file containing a column `filename` with names of files to skip.
        output_folder : str
            Where to store the output files.
        execution_time: str
            Datetime containing when the job was executed. Used to prefix the files name.
        """
        image_paths = get_image_paths(data_folder)

        logger.info(f"Number of input files found: {len(image_paths)}")

        if exclude_file != "":
            with open(os.path.join(data_folder, exclude_file), "r") as csv_file:
                reader = csv.reader(csv_file)
                _ = next(reader)
                exclude_list = {row[0] for row in reader}

                logger.info(f"Read {len(exclude_list)} rows from {exclude_file}")

                image_paths = [
                    img_path
                    for img_path in image_paths
                    if os.path.basename(img_path[1]) not in exclude_list
                ]

                logger.info(f"Number of input files remaining: {len(image_paths)}")

        if number_of_batches > len(image_paths):
            number_of_batches = (
                math.ceil(len(image_paths) / 50) if len(image_paths) > 50 else 1
            )
            logger.warning(
                f"Number of batches is greater than the number of images. Setting number_of_batches to {number_of_batches}."
            )

        images_per_batch = math.ceil(len(image_paths) / number_of_batches)

        for i in range(number_of_batches):
            start_index = i * images_per_batch
            end_index = min(start_index + images_per_batch, len(image_paths))

            batch_file_path = os.path.join(
                output_folder, f"{execution_time}_batch_{i}.txt"
            )
            with open(batch_file_path, "w") as batch_file:
                for j in range(start_index, end_index):
                    image_path = image_paths[j][1]
                    batch_file.write(
                        os.path.join(datastore_input_path, image_path) + "\n"
                    )
            logger.info(f"Batch {i} written to {batch_file_path}")
