import glob
import os
from pathlib import Path
from typing import Dict

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset


class YoloLabelsDataset(Dataset):
    def __init__(self, folder_path: str, image_area: int):
        self.folder_path = folder_path
        self.label_files = self.get_txt_files()
        self.image_area = image_area
        self._labels: Dict[str, npt.NDArray] = {}
        self._filtered_labels: Dict[str, npt.NDArray] = {}
        self._prepare_labels()

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, image_id: str):
        return self._labels[image_id]

    def get_labels(self):
        """
        Get dict with information about images and their corresponding labels.

        Returns
        -------

        """
        return self._labels

    def get_txt_files(self):
        txt_files = glob.glob(os.path.join(self.folder_path, "*.txt"))
        return [os.path.basename(file) for file in txt_files]

    def get_filtered_labels(self):
        return self._filtered_labels

    def _prepare_labels(self):
        """
        Loop through the yolo labels and store them in a dict.

        Each key in the dict is an image, each value is a ndarray (n_detections, 5)
        The 6 columns are in the yolo format, i.e. (target_class, x_c, y_c, width, height)

        Returns
        -------

        """
        self._labels = {}
        for file in self.label_files:
            with open(f"{self.folder_path}/{file}", "r") as f:
                lines = f.readlines()
            filename_no_extension = Path(os.path.splitext(file)[0]).stem
            self._labels[filename_no_extension] = np.array(
                [line.strip().split() for line in lines], dtype="f"
            )

        self._filtered_labels = self._labels.copy()

    def filter_by_size(self, size_to_keep):
        def _keep_labels_with_area_in_interval(bboxes, interval, img_area):
            product = bboxes[:, 3] * bboxes[:, 4] * img_area
            selected_rows = bboxes[
                (product >= interval[0]) & (product <= interval[1]), :
            ]

            return selected_rows

        for image_id, labels in self._filtered_labels.items():
            self._filtered_labels[image_id] = _keep_labels_with_area_in_interval(
                labels, size_to_keep, self.image_area
            )

        return self

    def filter_by_class(self, class_to_keep):
        def _keep_labels_with_class(array, class_id):
            return array[array[:, 0] == class_id, :]

        for image_id, labels in self._filtered_labels.items():
            self._filtered_labels[image_id] = _keep_labels_with_class(
                labels, class_id=class_to_keep
            )

        return self
