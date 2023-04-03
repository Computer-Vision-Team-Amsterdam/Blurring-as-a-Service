import itertools
import json
from os import listdir
from os.path import isfile, join
from typing import List

from blurring_as_a_service.performace_evaluation_pipeline.metrics.metrics_utils import (
    ImageSize,
)
from blurring_as_a_service.utils.bias_category_mapper import (
    BiasCategoryMapper,
    SensitiveCategories,
)


class FalseNegativeRateCalculator:
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
        self._get_gt_boxes_and_predicted_boxes_per_category(tagged_validation_folder)

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
                "| Category | Value | True Positives | TP Small | TP Medium | TP Large "
                "| False Negatives | FN Small | FN Medium | FN Large | False Negative Rate | False Negative Rate Small "
                "|False Negative Rate Medium | False Negative Rate Large|\n"
            )
            f.write(
                "| -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |"
                "-------------- | --------------- | ------------------ |\n"
            )
            for item in result:
                f.write(
                    f'| {item["category"]} | {item["value"]} | {item["true_positives"]} '
                    f'| {item["true_positives_small"]} | {item["true_positives_medium"]} | {item["true_positives_large"]} '
                    f'| {item["false_negatives"]} | {item["false_negatives_small"]} | {item["false_negatives_medium"]} '
                    f'| {item["false_negatives_large"]} | {item["false_negative_rate"]}'
                    f'| {item["false_negative_rate_small"]} | {item["false_negative_rate_medium"]}'
                    f'| {item["false_negative_rate_large"]}\n'
                )
            f.write(
                f"Thresholds used for these calculations: Small=`{ImageSize.small.value}`, Medium=`{ImageSize.medium.value}` "
                f"and Large=`{ImageSize.large.value}`."
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
        for category in SensitiveCategories().values:
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
        sensitive_category_options = list(
            {v[category_name] for k, v in self._statistics_per_category.items()}
        )

        metrics_for_all_sensitive_category_options = {
            metrics_for_sensitive_category_option: {
                "true_positives": 0,
                "true_positives_small": 0,
                "true_positives_medium": 0,
                "true_positives_large": 0,
                "false_negatives": 0,
                "false_negatives_small": 0,
                "false_negatives_medium": 0,
                "false_negatives_large": 0,
            }
            for metrics_for_sensitive_category_option in sensitive_category_options
        }
        for value in self._statistics_per_category.values():
            metrics_for_sensitive_category = metrics_for_all_sensitive_category_options[
                value[category_name]
            ]
            metrics_for_sensitive_category["true_positives"] += value["true_positives"]
            metrics_for_sensitive_category["false_negatives"] += value[
                "false_negatives"
            ]
            metrics_for_sensitive_category["true_positives_small"] += value[
                "true_positives_small"
            ]
            metrics_for_sensitive_category["true_positives_medium"] += value[
                "true_positives_medium"
            ]
            metrics_for_sensitive_category["true_positives_large"] += value[
                "true_positives_large"
            ]
            metrics_for_sensitive_category["false_negatives_small"] += value[
                "false_negatives_small"
            ]
            metrics_for_sensitive_category["false_negatives_medium"] += value[
                "false_negatives_medium"
            ]
            metrics_for_sensitive_category["false_negatives_large"] += value[
                "false_negatives_large"
            ]

        result = []
        for name, values in metrics_for_all_sensitive_category_options.items():
            true_positives = values["true_positives"]
            false_negatives = values["false_negatives"]
            true_positives_small = values["true_positives_small"]
            true_positives_medium = values["true_positives_medium"]
            true_positives_large = values["true_positives_large"]
            false_negatives_small = values["false_negatives_small"]
            false_negatives_medium = values["false_negatives_medium"]
            false_negatives_large = values["false_negatives_large"]

            false_negative_rate = (
                self._calculate_false_negative_rate(false_negatives, true_positives)
                if false_negatives + true_positives > 0
                else None
            )

            false_negative_rate_small = (
                self._calculate_false_negative_rate(
                    false_negatives_small, true_positives_small
                )
                if false_negatives_small + true_positives_small > 0
                else None
            )

            false_negative_rate_medium = (
                self._calculate_false_negative_rate(
                    false_negatives_medium, true_positives_medium
                )
                if false_negatives_medium + true_positives_medium > 0
                else None
            )

            false_negative_rate_large = (
                self._calculate_false_negative_rate(
                    false_negatives_large, true_positives_large
                )
                if false_negatives_large + true_positives_large > 0
                else None
            )

            result.append(
                {
                    "category": category_name,
                    "value": name,
                    "true_positives": true_positives,
                    "true_positives_small": true_positives_small,
                    "true_positives_medium": true_positives_medium,
                    "true_positives_large": true_positives_large,
                    "false_negatives": false_negatives,
                    "false_negatives_small": false_negatives_small,
                    "false_negatives_medium": false_negatives_medium,
                    "false_negatives_large": false_negatives_large,
                    "false_negative_rate": false_negative_rate,
                    "false_negative_rate_small": false_negative_rate_small,
                    "false_negative_rate_medium": false_negative_rate_medium,
                    "false_negative_rate_large": false_negative_rate_large,
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

        def _classify_bbox_area(bbox_area, image_size):
            for s in image_size:
                if s[0] <= bbox_area < s[1]:
                    return s

        for tagged_validation_file in self.get_all_filenames_in_dir(
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

                for i, (gt_box, gt_label) in enumerate(
                    zip(
                        tagged_validation_content["GT_boxes"],
                        tagged_validation_content["GT_labels"],
                    )
                ):
                    category_stats = self._statistics_per_category[gt_label]
                    category_stats["area"] = gt_box[2] * gt_box[3] * 32000000
                    size = _classify_bbox_area(category_stats["area"], ImageSize)

                    if tagged_validation_content["TP_labels"][i]:
                        category_stats["true_positives"] += 1
                        category_stats[f"true_positives_{size.name}"] += 1
                    else:
                        category_stats["false_negatives"] += 1
                        category_stats[f"false_negatives_{size.name}"] += 1

    def _get_gt_boxes_and_predicted_boxes_per_category(
        self, tagged_validation_folder: str
    ):
        """
        For each category of each file in tagged_validation_folder it gathers the ground truth boxes and all
         predicted boxes. This will be later used in computing the total blurred area per category.
        Args:
            tagged_validation_folder: Path to the folder containing all the tagged validation values.

        Returns:

        """
        for tagged_validation_file in self.get_all_filenames_in_dir(
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

                for i, (gt_box, gt_label) in enumerate(
                    zip(
                        tagged_validation_content["GT_boxes"],
                        tagged_validation_content["GT_labels"],
                    )
                ):
                    category_stats = self._statistics_per_category[gt_label]
                    category_stats["gt_boxes"].append(
                        [tagged_validation_content["GT_boxes"][i]]
                    )
                    category_stats["pred_boxes"].append(
                        tagged_validation_content["Pred_boxes"]
                        if "Pred_boxes" in tagged_validation_content
                        else []
                    )

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

        self._statistics_per_category = {}
        for d in all_grouped_categories:
            for attr_1, attr_2, attr_3 in [
                d["name"].split("/") + [""] * (3 - len(d["name"].split("/")))
            ]:
                self._statistics_per_category[d["id"]] = {
                    "id": d["id"],
                    "grouped_category": d["grouped_category"],
                    "true_positives": 0,
                    "true_positives_small": 0,
                    "true_positives_medium": 0,
                    "true_positives_large": 0,
                    "false_negatives": 0,
                    "false_negatives_small": 0,
                    "false_negatives_medium": 0,
                    "false_negatives_large": 0,
                    "area": 0,
                    "gt_boxes": [],
                    "pred_boxes": [],
                }
                if attr_1 == "licence_plate":
                    self._statistics_per_category[d["id"]].update(
                        {
                            "sex": "",
                            "age": "",
                            "skin_color": "",
                            "licence_plate_origin": attr_2,
                            "licence_plate_color": attr_3,
                        }
                    )
                else:
                    self._statistics_per_category[d["id"]].update(
                        {
                            "sex": attr_1,
                            "age": attr_2,
                            "skin_color": attr_3,
                            "licence_plate_origin": "",
                            "licence_plate_color": "",
                        }
                    )

    def get_areas(self):
        """
        Get absolute areas of all ground truth bounding boxes.
        Useful if we want to plot the distribution of areas.

        Returns
        -------

        """
        return [value["area"] for key, value in self._statistics_per_category.items()]

    @staticmethod
    def get_all_filenames_in_dir(directory_path: str) -> List[str]:
        return [f for f in listdir(directory_path) if isfile(join(directory_path, f))]

    @staticmethod
    def _calculate_false_negative_rate(false_negatives, true_positives):
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
        return round(false_negatives / (false_negatives + true_positives), 3)
