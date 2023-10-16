import sys
import glob
import os

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)
from yolov5.utils.dataloaders import IMG_FORMATS  # noqa: E402


def get_image_paths(input_container):
    """
    Get image paths, also searches in subdirectories.

    Parameters
    ----------
    input_container : str
        The path of the mounted root folder containing the images.

    Returns
    -------
    list of (str, str)
        A list of tuples, where each tuple contains the absolute file path
        and the relative file path to the input container for each image found.
    """
    image_paths = []
    pattern = os.path.join(input_container, "**/*.*")
    for file_path in glob.glob(pattern, recursive=True):
        _, extension = os.path.splitext(file_path)
        if extension.lstrip(".").lower() in IMG_FORMATS:
            image_paths.append((file_path, os.path.relpath(file_path, input_container)))
    return image_paths
