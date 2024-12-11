import logging
import math
import os

logger = logging.getLogger(__name__)

from blurring_as_a_service.pre_inference_pipeline.source.image_paths import (  # noqa: E402
    get_image_paths,
)


class WorkloadSplitter:
    @staticmethod
    def create_batches(data_folder, datastore_input_path, number_of_batches, output_folder, execution_time):
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
        output_folder : str
            Where to store the output files.
        execution_time: str
            Datetime containing when the job was executed. Used to prefix the files name.
        """
        # Get all image paths within the input folder
        image_paths = get_image_paths(data_folder)

        # Ensure number_of_batches is not greater than the number of images
        if number_of_batches > len(image_paths):
            logger.warning(
                "Number of batches is greater than the number of images. Setting number_of_batches to 1."
            )
            number_of_batches = 1

        images_per_batch = math.ceil(len(image_paths) / number_of_batches)

        # Process images and create batches
        for i in range(number_of_batches):
            start_index = i * images_per_batch
            end_index = min(start_index + images_per_batch, len(image_paths))

            with open(
                f"{output_folder}/{execution_time}_batch_{i}.txt", "w"
            ) as batch_file:
                for j in range(start_index, end_index):
                    # Only get the relative image paths
                    image_path = image_paths[j][1]
                    batch_file.write(os.path.join(datastore_input_path, image_path) + "\n")
