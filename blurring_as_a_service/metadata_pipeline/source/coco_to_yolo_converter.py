import json
from typing import List


class CocoToYoloConverter:
    """
    Converts a COCO annotation dataset to a YOLOv5 format.
    - The bbox changes from x_min, y_min, width, height to x_center, y_center, width, height.
    - Categories will be grouped into two main cateogires 0=person and 1=license plate.
    - It generates one file per image.
    """

    def __init__(self, coco_file: str, yolo_folder: str):
        """
        Parameters
        ----------
        coco_file
            path to input coco file
        yolo_folder
            path to folder where to store all .txt files
        """
        self._output_dir = yolo_folder

        with open(coco_file) as f:
            self._input = json.load(f)

        self._grouped_categories = self._group_categories(self._input["categories"])

    @staticmethod
    def _group_categories(categories: List[dict]) -> List[dict]:
        """
        Group categories into person or licence plates based on the original category name.

        Parameters
        ----------
        categories
            Original category name, example:
                [{"id": 1, "name": "man"}, {"id": 2, "name": "man/child"}, {"id": 3, "name": "man/child/light"}]

        Returns
        -------
            Grouped category, example: [
                {"id": 1, "name": "man", 'grouped_category': 0},
                {"id": 2, "name": "man/child", 'grouped_category': 0},
                {"id": 3, "name": "man/child/light", 'grouped_category': 0}
            ]
        """
        for category in categories:
            if category["name"].startswith("licence_plate"):
                category["grouped_category"] = 1
            else:
                category["grouped_category"] = 0
        return categories

    def _get_grouped_category(self, category_id) -> int:
        """
        Returns the correct grouped category for a specific annotation.

        Parameters
        ----------
        category_id:
            Original category.

        Returns
        -------
        Grouped category.

        """
        return list(
            filter(lambda cat: cat["id"] == category_id, self._grouped_categories)
        )[0]["grouped_category"]

    def _write_to_txt(self, image_name, per_image_annotations):
        """
        Writes the original COCO file parsed into YOLO format.

        Each line contains: [grouped_category, x, ,y, width, height, category]
        For the details about group category look at _group_categories method.

        Parameters
        ----------
        image_name:
            Name of the image.
        per_image_annotations:
            Annotations in COCO format.
        """
        lines = []
        for annotation in per_image_annotations:
            x, y, width, height = annotation["bbox"]
            xc = x + width / 2
            yc = y + height / 2
            grouped_category = self._get_grouped_category(annotation["category_id"])
            line = f'{grouped_category} {xc} {yc} {width} {height} {annotation["category_id"]}'
            lines.append(line)

        with open(f"{self._output_dir}/{image_name}.txt", "w") as f:
            f.write("\n".join(lines))

    def convert(self):
        """
        Converts a COCO annotation dataset to a YOLOv5 format.
        - The bbox changes from x_min, y_min, width, height to x_center, y_center, width, height.
        - Categories will be grouped into two main cateogires 0=person and 1=license plate.
        - It generates one file per image.
        """
        for i, image in enumerate(self._input["images"]):
            # collect corresponding annotations
            annotations = [
                annotation
                for annotation in self._input["annotations"]
                if annotation["image_id"] == image["id"]
            ]

            # write annotations to txt file. image_name is TMXblabla.jpg
            self._write_to_txt(
                image_name=image["file_name"].split("/")[-1].split(".")[0],
                per_image_annotations=annotations,
            )
