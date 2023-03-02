import itertools
import json
from enum import Enum
from os import listdir
from os.path import isfile, join
from typing import List

import matplotlib.pyplot as plt
from tqdm import tqdm

from blurring_as_a_service.metrics.metrics_utils import generate_binary_mask
from blurring_as_a_service.metrics.total_blurred_area import TotalBlurredArea
from blurring_as_a_service.metrics.yolo_labels_dataset import YoloLabelsDataset
from blurring_as_a_service.utils.bias_category_mapper import BiasCategoryMapper


class Size(Enum):
    small = [0, 5000]
    medium = [5000, 10000]
    large = [10000, 1000000]


class Class(Enum):
    person = 0
    licence_plate = 1


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
                f"Thresholds used for these calculations: Small=`{Size.small.value}`, Medium=`{Size.medium.value}` "
                f"and Large=`{Size.large.value}`."
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
        for category in [
            "grouped_category",
            "sex",
            "age",
            "skin_color",
            "licence_plate_origin",
            "licence_plate_color",
        ]:
            print(category)
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
                "true_positives_small": 0,
                "true_positives_medium": 0,
                "true_positives_large": 0,
                "false_negatives": 0,
                "false_negatives_small": 0,
                "false_negatives_medium": 0,
                "false_negatives_large": 0,
            }
            for available_category in available_categories
        }
        for value in self._statistics_per_category.values():
            category = all_categories[value[category_name]]
            category["true_positives"] += value["true_positives"]
            category["false_negatives"] += value["false_negatives"]
            category["true_positives_small"] += value["true_positives_small"]
            category["true_positives_medium"] += value["true_positives_medium"]
            category["true_positives_large"] += value["true_positives_large"]
            category["false_negatives_small"] += value["false_negatives_small"]
            category["false_negatives_medium"] += value["false_negatives_medium"]
            category["false_negatives_large"] += value["false_negatives_large"]

        result = []
        for name, values in all_categories.items():
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
                    bbox_area = gt_box[2] * gt_box[3] * 32000000
                    category_stats["area"] = bbox_area
                    if tagged_validation_content["TP_labels"][i]:
                        category_stats["true_positives"] += 1
                        if Size.small.value[0] <= bbox_area < Size.small.value[1]:
                            category_stats["true_positives_small"] += 1
                        if Size.medium.value[0] <= bbox_area < Size.medium.value[1]:
                            category_stats["true_positives_medium"] += 1
                        if Size.large.value[0] <= bbox_area < Size.large.value[1]:
                            category_stats["true_positives_large"] += 1
                    else:
                        category_stats["false_negatives"] += 1
                        if Size.small.value[0] <= bbox_area < Size.small.value[1]:
                            category_stats["false_negatives_small"] += 1
                        if Size.medium.value[0] <= bbox_area < Size.medium.value[1]:
                            category_stats["false_negatives_medium"] += 1
                        if Size.large.value[0] <= bbox_area < Size.large.value[1]:
                            category_stats["false_negatives_large"] += 1

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
                if attr_1 == "licence_plate":
                    self._statistics_per_category[d["id"]] = {
                        "id": d["id"],
                        "grouped_category": d["grouped_category"],
                        "sex": "",
                        "age": "",
                        "skin_color": "",
                        "licence_plate_origin": attr_2,
                        "licence_plate_color": attr_3,
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
                else:
                    self._statistics_per_category[d["id"]] = {
                        "id": d["id"],
                        "grouped_category": d["grouped_category"],
                        "sex": attr_1,
                        "age": attr_2,
                        "skin_color": attr_3,
                        "licence_plate_origin": "",
                        "licence_plate_color": "",
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

    def get_areas(self):
        return [value["area"] for key, value in self._statistics_per_category.items()]

    def _plot_area_distribution(self, path: str, plot_name: str):
        plt.hist(self.get_areas(), bins=50, range=[0, 25000], color="blue", alpha=0.5)
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of bbox area of {plot_name}")
        plt.savefig(f"{path}/{plot_name}.jpg")
        plt.show()

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


def get_total_blurred_area_statistics(true_labels, predicted_lables):
    total_blurred_area = TotalBlurredArea()

    for image_id in tqdm(true_labels.keys(), total=len(true_labels)):
        tba_true_mask = generate_binary_mask(true_labels[image_id][:, 1:5])
        tba_pred_mask = generate_binary_mask(predicted_lables[image_id][:, 1:5])

        total_blurred_area.update_statistics_based_on_masks(
            true_mask=tba_true_mask, predicted_mask=tba_pred_mask
        )

    results = total_blurred_area.get_statistics()

    return results


def collect_tba_results_per_class_and_size(true_path, pred_path):
    predicted_dataset = YoloLabelsDataset(folder_path=pred_path)
    results = {}

    # ====== PERSONS ====== #

    for size in Size:
        true_persons_size = (
            YoloLabelsDataset(folder_path=true_path)
            .filter_by_class(class_to_keep=Class.person.value)
            .filter_by_size(size_to_keep=size.value)
            .get_filtered_labels()
        )
        results[f"persons_{size.name}"] = get_total_blurred_area_statistics(
            true_persons_size, predicted_dataset.get_labels()
        )

    # ====== LICENCE PLATE ====== #

    for size in Size:
        true_licences_size = (
            YoloLabelsDataset(folder_path=true_path)
            .filter_by_class(class_to_keep=Class.licence_plate.value)
            .filter_by_size(size_to_keep=size.value)
            .get_filtered_labels()
        )
        results[f"licences_{size.name}"] = get_total_blurred_area_statistics(
            true_licences_size, predicted_dataset.get_labels()
        )

    return results


def store_tba_result(results, markdown_output_path="tba_scores.mda"):
    with open(markdown_output_path, "w") as f:
        f.write(
            " Person Small | Person Medium | Person Large |"
            " License Plate Small |  License Plate Medium  | License Plate Large |\n"
        )
        f.write("|----- | ----- |  ----- | ----- | ----- | ----- |\n")
        f.write(
            f'| {results["persons_small"]["recall"]} | {results["persons_medium"]["recall"]} '
            f'| {results["persons_large"]["recall"]}| {results["licences_small"]["recall"]} '
            f'| {results["licences_medium"]["recall"]} | {results["licences_large"]["recall"]}|\n'
        )

        f.write(
            f"Thresholds used for these calculations: Small=`{Size.small.value}`, Medium=`{Size.medium.value}` "
            f"and Large=`{Size.large.value}`."
        )
        f.write(
            f"Thresholds used for these calculations: Small=`{Size.small.value}`, Medium=`{Size.medium.value}` "
            f"and Large=`{Size.large.value}`."
        )


def collect_and_store_tba_results_per_class_and_size(
    ground_truth_path, predictions_path, markdown_output_path
):
    results = collect_tba_results_per_class_and_size(
        ground_truth_path, predictions_path
    )
    store_tba_result(results, markdown_output_path)
