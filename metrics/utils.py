from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageOps

import matplotlib.pyplot as plt


def parse_labels(file_path: str) -> Tuple[List[int], List[Tuple[float, float, float, float]]]:
    """
     Parses a labels file with the following normalized format: [x_center, y_center, width, height]

    :param file_path: The path to the labels file to be parsed.
    :return: A tuple with two lists: classes and bounding_boxes
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    f.close()

    classes = []
    bounding_boxes = []
    for line in lines:
        elements = line.strip().split()
        classes.append(int(elements[0]))
        bounding_boxes.append((float(elements[1]), float(elements[2]), float(elements[3]), float(elements[4])))
    return classes, bounding_boxes


def generate_mask_binary(*args, **kwargs):
    mask = generate_mask(*args, **kwargs)
    mask_array = np.array(mask)
    mask_2d = mask_array.sum(axis=2) > 0
    mask_binary = mask_2d.astype(np.uint8)
    return mask_binary


def generate_mask(bounding_boxes: List[Tuple[float, float, float, float]], image: Image) -> Image:
    """
       Generates a mask for an image given a list of bounding boxes.

       Parameters:
        :param bounding_boxes: (List[Tuple[float, float, float, float]]): List of bounding boxes with normalised
                                                                     (x_center, y_center, width, height)
        :param image: (PIL.Image): An image to generate the mask for.


        :return: PIL.Image: The generated mask image.
       """
    mask = np.zeros_like(np.array(image))
    for bounding_box in bounding_boxes:
        x_center, y_center, width, height = bounding_box
        x_min = int((x_center - width / 2) * image.width)
        y_min = int((y_center - height / 2) * image.height)
        x_max = int((x_center + width / 2) * image.width)
        y_max = int((y_center + height / 2) * image.height)
        mask[y_min:y_max, x_min:x_max, :] = np.array(image)[y_min:y_max, x_min:x_max, :]
    return Image.fromarray(mask)


def visualize_mask(mask, image):
    plt.imshow(np.array(image))
    plt.imshow(np.array(mask), alpha=0.8)
    plt.savefig("mask.jpg", dpi=500)
    plt.show()
