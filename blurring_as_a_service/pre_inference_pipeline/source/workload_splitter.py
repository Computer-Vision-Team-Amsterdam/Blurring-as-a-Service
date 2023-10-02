import math
import os
import sys
import glob

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)
from yolov5.utils.dataloaders import IMG_FORMATS  # noqa: E402


# Function to get all paths to image files within a directory and its subfolders
def get_image_paths(directory, formats):
    image_paths = []
    pattern = os.path.join(directory, "**/*.*")
    for file_path in glob.glob(pattern, recursive=True):
        _, extension = os.path.splitext(file_path)
        if extension.lstrip(".").lower() in formats:
            image_paths.append(os.path.relpath(file_path, directory))
    return image_paths


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
        # Get all image paths within the input folder
        image_files = get_image_paths(data_folder, IMG_FORMATS)

        images_per_batch = math.ceil(len(image_files) / number_of_batches)

        # Process images and create batches
        for i in range(number_of_batches):
            start_index = i * images_per_batch
            end_index = min(start_index + images_per_batch, len(image_files))

            with open(
                    f"{output_folder}/{execution_time}_batch_{i}.txt", "w"
            ) as batch_file:
                for j in range(start_index, end_index):
                    image_path = image_files[j]
                    batch_file.write(image_path + "\n")
