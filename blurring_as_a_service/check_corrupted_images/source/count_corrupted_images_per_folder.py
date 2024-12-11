import glob
import os
import sys
from collections import defaultdict

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)
# Add the yolov5 path to sys.path
sys.path.append(yolov5_path)
from yolov5.utils.dataloaders import IMG_FORMATS  # noqa: E402


def count_corrupted_images_per_folder(input_container):
    image_counts = defaultdict(lambda: defaultdict(int))
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
    validation_code = 0
    statfile = os.stat(filename)
    filesize = statfile.st_size
    if filesize == 0:
        validation_code = 1
    else:
        with open(filename, 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            validation_code = 2
    return validation_code
