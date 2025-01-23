import glob
import os
from typing import List, Tuple

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"


def get_image_paths(input_container: str) -> List[Tuple[str, str]]:
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
    pattern = os.path.join(input_container, "**/*.*")
    image_paths = [
        (file_path, os.path.relpath(file_path, input_container))
        for file_path in glob.glob(pattern, recursive=True)
        if os.path.splitext(file_path)[1].lstrip(".").lower() in IMG_FORMATS
    ]
    return image_paths
