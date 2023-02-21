from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType


def process_image_labels(labels):

    true_classes, true_bboxes = parse_labels(labels["true"])
    pred_classes, pred_bboxes = parse_labels(labels["predicted"])

    tba_true_mask = generate_binary_mask(true_bboxes)
    tba_pred_mask = generate_binary_mask(pred_bboxes)

    # discard true and pred classes which are licence_plates
    person_true_bboxes_filtered = [
        true_bboxes[i] for i in range(len(true_bboxes)) if true_classes[i] == 0
    ]
    person_pred_bboxes_filtered = [
        pred_bboxes[i] for i in range(len(pred_bboxes)) if pred_classes[i] == 0
    ]

    uba_true_mask = generate_binary_mask(
        person_true_bboxes_filtered, consider_upper_half=True
    )
    uba_pred_mask = generate_binary_mask(
        person_pred_bboxes_filtered, consider_upper_half=True
    )

    return tba_true_mask, tba_pred_mask, uba_true_mask, uba_pred_mask


def parse_labels(
    file_path: str,
) -> Tuple[List[int], List[Tuple[float, float, float, float]]]:
    """
     Parses a labels file with the following normalized format: [x_center, y_center, width, height]

    :param file_path: The path to the labels file to be parsed.
    :return: A tuple with two lists: classes and bounding_boxes
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    f.close()

    classes = [int(line.strip().split()[0]) for line in lines]
    bounding_boxes = [
        (
            float(line.strip().split()[1]),
            float(line.strip().split()[2]),
            float(line.strip().split()[3]),
            float(line.strip().split()[4]),
        )
        for line in lines
    ]
    return classes, bounding_boxes


def generate_binary_mask(
    bounding_boxes, image_width=8000, image_height=4000, consider_upper_half=False
):
    """

    :param consider_upper_half:
    :param image_height:
    :param image_width:
    :param bounding_boxes:
    :return:
    """

    mask = np.zeros((image_height, image_width))

    if len(bounding_boxes):
        bounding_boxes = np.array(bounding_boxes)
        y_min = (
            (bounding_boxes[:, 1] - bounding_boxes[:, 3] / 2) * image_height
        ).astype(int)
        x_min = (
            (bounding_boxes[:, 0] - bounding_boxes[:, 2] / 2) * image_width
        ).astype(int)
        x_max = (
            (bounding_boxes[:, 0] + bounding_boxes[:, 2] / 2) * image_width
        ).astype(int)
        if consider_upper_half:
            y_max = (bounding_boxes[:, 1] * image_height).astype(int)
        else:
            y_max = (
                (bounding_boxes[:, 1] + bounding_boxes[:, 3] / 2) * image_height
            ).astype(int)
        for i in range(len(x_min)):
            mask[y_min[i] : y_max[i], x_min[i] : x_max[i]] = 1

    return mask


def generate_mask(
    bounding_boxes, image: ImageType, consider_upper_half=False
) -> ImageType:
    """
    Generates a mask for an image given a list of bounding boxes.

    Parameters:
     :param bounding_boxes: (List[Tuple[float, float, float, float]]): List of bounding boxes with normalised
                                                                  (x_center, y_center, width, height)
     :param image: (PIL.Image): An image to generate the mask for.


     :return: PIL.Image: The generated mask image.
    """
    mask = np.zeros_like(np.array(image))

    if len(bounding_boxes):
        bounding_boxes = np.array(bounding_boxes)
        y_min = (
            (bounding_boxes[:, 1] - bounding_boxes[:, 3] / 2) * image.height
        ).astype(int)
        x_min = (
            (bounding_boxes[:, 0] - bounding_boxes[:, 2] / 2) * image.width
        ).astype(int)
        x_max = (
            (bounding_boxes[:, 0] + bounding_boxes[:, 2] / 2) * image.width
        ).astype(int)
        if consider_upper_half:
            y_max = (bounding_boxes[:, 1] * image.height).astype(int)
        else:
            y_max = (
                (bounding_boxes[:, 1] + bounding_boxes[:, 3] / 2) * image.height
            ).astype(int)
        for i in range(len(x_min)):
            mask[y_min[i] : y_max[i], x_min[i] : x_max[i], :] = np.array(image)[
                y_min[i] : y_max[i], x_min[i] : x_max[i], :
            ]
    return Image.fromarray(mask)


def visualize_mask(mask, image):
    plt.imshow(np.array(image))
    plt.imshow(np.array(mask), alpha=0.8)
    plt.savefig("mask.jpg", dpi=500)
    plt.show()
