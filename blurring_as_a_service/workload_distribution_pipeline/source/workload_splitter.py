import math
import os
from datetime import datetime

IMG_FORMATS = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')  # image suffixes defined in YOLOv5

class WorkloadSplitter:
    @staticmethod
    def create_batches(data_folder, date_folders, number_of_batches, output_folder):
        """
        Starting from a data folder and a list of date folders, iterates over all image files in the data folder and equally
        groups them into number_of_batches batches. These groups are stored into multiple txt files where each line is a file
        including the relative path.

        Parameters
        ----------
        data_folder : str
            Root folder containing the images.
        date_folders : list of str
            List of folder names with date format (YYYY-MM-DD).
        number_of_batches : int
            Number of files to distribute the data.
        output_folder : str
            Where to store the output files.
        """
        def validate_date_format(date_string):
            try:
                datetime.strptime(date_string, '%Y-%m-%d')
            except ValueError as e:
                error_msg = f"Invalid date format: {date_string}. Expected format: YYYY-MM-DD"
                print(error_msg)
                raise e

        # Validate the format of all date folders
        for date_folder in date_folders:
            validate_date_format(date_folder)

        image_files = []
        # Collect image files from date folders
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.lower().endswith(IMG_FORMATS):
                    file_folder = os.path.basename(os.path.dirname(os.path.join(root, file)))
                    # Check if the file's parent folder is in date_folders
                    if file_folder in date_folders:
                        relative_path = os.path.relpath(os.path.join(root, file), data_folder)
                        image_files.append(relative_path)

        images_per_batch = math.ceil(len(image_files) / number_of_batches)

        # Process images and create batches
        for i in range(number_of_batches):
            start_index = i * images_per_batch
            end_index = min(start_index + images_per_batch, len(image_files))

            with open(f"{output_folder}/batch_{i}.txt", "w") as batch_file:
                for j in range(start_index, end_index):
                    batch_file.write(image_files[j] + "\n")
