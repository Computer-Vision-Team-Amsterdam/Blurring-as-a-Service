import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


class DataStatistics:
    """
    Compute and visualize different statistics from the data-prep
    """

    def __init__(self, json_file: Path, output_dir: Optional[str] = None) -> None:
        """
        json file can be either COCO annotation or COCO results file
        """
        with open(json_file) as f:
            self.data = json.load(f)
        f.close()

        self.output_dir = output_dir
        self.widths, self.heights = self.collect_dimensions(self.data)
        self.areas = [
            width * height for width, height in zip(self.widths, self.heights)
        ]

    def collect_dimensions(self, data: Any) -> Tuple[List[int], List[int]]:
        """
        Collects widths and heights from json file
        """
        # assert if we have a list/results json or a dict/annotations json
        if isinstance(data, list):  # this is a results json file
            pass
        if isinstance(data, dict):  # this is an annotation json file
            data = data["annotations"]

        widths = []
        heights = []
        for ann in data:
            width = ann["bbox"][2]
            height = ann["bbox"][3]

            widths.append(int(width))
            heights.append(int(height))

        return widths, heights

    def plot_dimensions_distribution(self, plot_name: str) -> None:
        """
        Scatter plot with height and widths of containers

         plot_name : includes the file type, i.e. jpg
        """
        if self.output_dir is None:
            raise ValueError("output_dir cannot be None")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.scatter(self.widths, self.heights, alpha=0.5)
        plt.xlabel("width")
        plt.ylabel("height")
        plt.savefig(Path(self.output_dir, plot_name))

    def plot_areas_distribution(self, plot_name: str) -> None:
        """
        Histogram with containers areas
        plot_name : includes the file type, i.e. jpg
        """

        if self.output_dir is None:
            raise ValueError("output_dir cannot be None")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.hist(self.areas, bins=30, range=(0, 50000))
        plt.xlabel("Container bbox area")
        plt.ylabel("Count")
        plt.savefig(Path(self.output_dir, plot_name))

    def update(self, data: List[Dict[str, Any]]) -> None:
        """
        Update object with new data-prep and recalcute its statistics
        :param data:
        :return:
        """
        self.data = data
        self.widths, self.heights = self.collect_dimensions(self.data)
        self.areas = [
            width * height for width, height in zip(self.widths, self.heights)
        ]
