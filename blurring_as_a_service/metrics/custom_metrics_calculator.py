import itertools
import json
from os import listdir
from os.path import isfile, join
from typing import List

from blurring_as_a_service.utils.bias_category_mapper import BiasCategoryMapper


class CustomMetricsCalculator:
    """
    Given a folder containing tagged_validation results for multiple images, it loads all the results,
    groups the true positives and false negatives per label, and calculates the metrics.

    The content of the tagged-validation for one image will be:
        - GT_boxes: List[List[float] ground truth boxes, i.e. the coordinates of the ground truth labels
        - GT_labels: List[int] ground truth labels in interval [0, 69]
        - TP_labels: List[boolean/int(binary)] true positives.
            - TP_labels[i] is True/1 if we have a true positive for GT_labels[i].
            - TP_labels[i] is False/0 if we have a false negative for GT_labels[i].

    """

    def __init__(self, tagged_validation_folder, coco_file_with_categories):
        """
        Parameters
        ----------
        tagged_validation_folder
            Folder containing yolo tagged validation files.
        coco_file_with_categories
            File containing the categories.
        """
        self._get_and_prepare_categories(coco_file_with_categories)
        self._calculate_true_positives_and_false_negatives_per_category(
            tagged_validation_folder
        )

    def calculate_and_store_metrics(self, markdown_output_path):
        """
        Calculates the custom metrics for all the categories and stores them in a markdown file.

        Parameters
        ----------
        markdown_output_path
            Path where to store the markdown file.

        """
        result = self.calculate_false_negative_rate_for_all_categories()

        with open(markdown_output_path, "w") as f:
            f.write(
                "| Category | Value | True Positives | False Negatives | False Negative Rate |\n"
            )
            f.write(
                "| -------- | ----- | -------------- | --------------- | ------------------ |\n"
            )
            for item in result:
                f.write(
                    f'| {item["category"]} | {item["value"]} | {item["true_positives"]} '
                    f'| {item["false_negatives"]} | {item["false_negative_rate"]} |\n'
                )

    def calculate_false_negative_rate_for_all_categories(self) -> List[dict]:
        """
        Using the calculate_false_negative_rate_for_a_category function, it calculates the value for the following categories:
            - 'grouped_category': Persons or licence plates.
            - 'sex'.
            - 'age'.
            - 'skin_color'.
        Returns
        -------
        A list where each element is a category and its metrics, included the false negative rate.
        """
        result = []
        for category in ["grouped_category", "sex", "age", "skin_color"]:
            result.append(self.calculate_false_negative_rate_for_a_category(category))
        return list(itertools.chain(*result))

    def calculate_false_negative_rate_for_a_category(
        self, category_name: str
    ) -> List[dict]:
        """
        Given a category name it calculates the false negative rate of that category
        checking the values in _statistics_per_category.

        Parameters
        ----------
        category_name

        Returns
        -------
        A list where each element is a category value and its metrics, included the false negative rate.
        """
        available_categories = list(
            {v[category_name] for k, v in self._statistics_per_category.items()}
        )
        all_categories = {
            available_category: {
                "true_positives": 0,
                "false_negatives": 0,
            }
            for available_category in available_categories
        }
        for value in self._statistics_per_category.values():
            category = all_categories[value[category_name]]
            category["true_positives"] += value["true_positives"]
            category["false_negatives"] += value["false_negatives"]

        result = []
        for name, values in all_categories.items():
            true_positives = values["true_positives"]
            false_negatives = values["false_negatives"]
            if false_negatives + true_positives > 0:
                false_negative_rate = self._calculate_false_negative_rate(
                    false_negatives, true_positives
                )
            else:
                false_negative_rate = None
            result.append(
                {
                    "category": category_name,
                    "value": name,
                    "true_positives": true_positives,
                    "false_negatives": false_negatives,
                    "false_negative_rate": false_negative_rate,
                }
            )
        return result

    def _calculate_true_positives_and_false_negatives_per_category(
        self, tagged_validation_folder: str
    ):
        """
        For each category of each file in tagged_validation_folder it calculates the true positives and the false negatives.

        Parameters
        ----------
        tagged_validation_folder
            Path to the folder containing all the tagged validation values.
        Raises
        ------
        Exception in case the len of GT_boxes, GT_labels and TP_labels are not matching.

        """
        for tagged_validation_file in self._get_all_filenames_in_dir(
            tagged_validation_folder
        ):
            with open(f"{tagged_validation_folder}/{tagged_validation_file}") as f:
                tagged_validation_content = json.load(f)
                if any(
                    len(tagged_validation_content[key])
                    != len(tagged_validation_content["GT_labels"])
                    for key in ["GT_boxes", "TP_labels"]
                ):
                    raise Exception(
                        f"{tagged_validation_file} not well formed, "
                        f"the len of GT_boxes, GT_labels and TP_labels is not matching."
                    )
                for i, gt_label in enumerate(tagged_validation_content["GT_labels"]):
                    category_stats = self._statistics_per_category[gt_label]
                    category_stats[
                        "true_positives"
                        if tagged_validation_content["TP_labels"][i]
                        else "false_negatives"
                    ] += 1

    def _get_and_prepare_categories(self, coco_file_with_categories):
        """
        Retrieves all the possible categories and values from the coco file and creates the _statistics_per_category
        attribute that is used to generate the metrics.

        Parameters
        ----------
        coco_file_with_categories

        Returns
        -------

        """
        with open(coco_file_with_categories) as f:
            categories_file_content = json.load(f)
        self._bias_category_mapper = BiasCategoryMapper(
            categories_file_content["categories"]
        )
        all_grouped_categories = self._bias_category_mapper.get_all_grouped_categories()
        self._statistics_per_category = {
            d["id"]: {
                "id": d["id"],
                "grouped_category": d["grouped_category"],
                "sex": sex,
                "age": age,
                "skin_color": skin_color,
                "true_positives": 0,
                "false_negatives": 0,
            }
            for d in all_grouped_categories
            for sex, age, skin_color in [
                d["name"].split("/") + [""] * (3 - len(d["name"].split("/")))
            ]
        }

    @staticmethod
    def _get_all_filenames_in_dir(directory_path: str) -> List[str]:
        return [f for f in listdir(directory_path) if isfile(join(directory_path, f))]

    @staticmethod
    def _calculate_false_negative_rate(false_negatives: int, true_positives: int):
        """
        Calculates the False Negative Rate metric.
        Finding no disparities in this metric means that we would have the same chance of not blurring someone
        across the groups.

        Parameters
        ----------
        false_negatives
            Number of false negatives.
        true_positives
            Number of true positives.

        Returns
        -------
        False negative rate.
        """
        return false_negatives / (false_negatives + true_positives)
