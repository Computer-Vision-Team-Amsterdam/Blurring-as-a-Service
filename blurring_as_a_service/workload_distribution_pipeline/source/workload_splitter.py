import math
import os


class WorkloadSplitter:
    @staticmethod
    def create_batches(data_folder, number_of_batches, output_folder):
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
        data_folder
            Folder containing the images.
        number_of_batches
            Number of files to distribute the data.
        output_folder
            Where to store the output files.
        """
        image_files = []
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg"):
                    image_files.append(os.path.join(root, file))

        images_per_batch = math.ceil(len(image_files) / number_of_batches)

        for i in range(number_of_batches):
            start_index = i * images_per_batch
            end_index = min(start_index + images_per_batch, len(image_files))

            with open(f"{output_folder}/batch_{i}.txt", "w") as batch_file:
                for j in range(start_index, end_index):
                    relative_path = os.path.relpath(image_files[j], data_folder)
                    batch_file.write(relative_path + "\n")