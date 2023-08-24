from enum import Enum
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType


class ImageSize(Enum):
    small = [0, 5000]
    medium = [5000, 10000]
    large = [10000, 32000000]

    def __repr__(self):
        return self.value

    def __getitem__(self, index):
        return self.value[index]


class TargetClass(Enum):
    person = 0
    license_plate = 1

    def __repr__(self):
        return self.value


def parse_labels(
    file_path: str,
) -> Tuple[List[int], List[Tuple[float, float, float, float]]]:
    """
     Parses a txt file with the following normalized format: [class x_center, y_center, width, height]

    Parameters
    ----------
    file_path The path to the labels file to be parsed.

    Returns (classes and bounding_boxes)
    -------

    """
    with open(file_path, "r") as f:
        lines = f.readlines()

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
    Creates binary mask where all points inside the bounding boxes are 1, 0 otherwise.

    Parameters
    ----------
    bounding_boxes: list of bounding box coordinates
    image_width
    image_height
    consider_upper_half: only look at the upper half of the bounding boxes

    Returns
    -------

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
    Generates an RGB mask for an image given a list of bounding boxes.
    Similar to generate_binary_mask, but this is used for visualizations, mainly as sanity checks

    Parameters
    ----------
    bounding_boxes List of bounding boxes with normalised (x_center, y_center, width, height)
    image  An image to generate the mask for.
    consider_upper_half only look at the upper half of the bounding boxes

    Returns
    -------

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


def visualize_mask(image, name="mask.jpg"):
    plt.figure(name, figsize=(16, 8))
    plt.imshow(np.array(image))
    plt.savefig(name, dpi=500)
    plt.show()
