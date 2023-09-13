import math
import os

from blurring_as_a_service.utils.generics import IMG_FORMATS  # noqa: E402


class WorkloadSplitter:
    @staticmethod
    def create_batches(data_folder, number_of_batches, output_folder, execution_time):
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
        image_files = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith(IMG_FORMATS):
                    image_files.append(os.path.join(root, file))

        images_per_batch = math.ceil(len(image_files) / number_of_batches)

        # Process images and create batches
        for i in range(number_of_batches):
            start_index = i * images_per_batch
            end_index = min(start_index + images_per_batch, len(image_files))

            with open(
                f"{output_folder}/{execution_time}_batch_{i}.txt", "w"
            ) as batch_file:
                for j in range(start_index, end_index):
                    file_name = os.path.basename(image_files[j])
                    image_path = os.path.join(execution_time, file_name)
                    batch_file.write(image_path + "\n")
