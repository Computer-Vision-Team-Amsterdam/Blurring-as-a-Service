import os
from collections import defaultdict

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"


def count_corrupted_images_per_folder(input_container: str) -> defaultdict:
    """
    Count the number of corrupted images in each folder within the input container.

    Parameters
    ----------
    input_container : str
        The path to the input container directory.

    Returns
    -------
    defaultdict
        A nested defaultdict where the keys are folder paths and the values are dictionaries
        with counts of total, good, empty, and corrupted images.
    """
    image_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for root, _, files in os.walk(input_container):
        total_images = good_images = empty_images = corrupted_images = 0
        for file in files:
            if file.lower().endswith(IMG_FORMATS):
                validation_code = check_image(os.path.join(root, file))
                total_images += 1
                if validation_code == 0:
                    good_images += 1
                elif validation_code == 1:
                    empty_images += 1
                elif validation_code == 2:
                    corrupted_images += 1

        if total_images > 0:
            image_counts[root]["total_images"] += total_images
        if good_images > 0:
            image_counts[root]["good_images"] += good_images
        if empty_images > 0:
            image_counts[root]["empty_images"] += empty_images
        if corrupted_images > 0:
            image_counts[root]["corrupted_images"] += corrupted_images

    return image_counts


def check_image(filename: str) -> int:
    """
    Checks if an image file is corrupted based on its size and end-of-file marker.

    Parameters
    ----------
    filename : str
        The path to the image file to be checked.

    Returns
    -------
    int
        A validation code indicating the status of the image file:
        - 0: The image file is valid.
        - 1: The image file is empty.
        - 2: The image file does not end with the JPEG end-of-image marker (0xFFD9).
    """
    validation_code = 0
    statfile = os.stat(filename)
    filesize = statfile.st_size
    if filesize == 0:
        validation_code = 1
    else:
        with open(filename, "rb") as f:
            check_chars = f.read()[-2:]
        if (
            check_chars != b"\xff\xd9"
        ):  # Check if the last two bytes are the JPEG end-of-image marker
            validation_code = 2
    return validation_code
