import logging
import os

logger = logging.getLogger(__name__)

IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
)  # include image suffixes


def find_image_paths(root_folder):
    image_paths = []
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in IMG_FORMATS):
                image_path = os.path.join(foldername, filename)
                image_paths.append(image_path)
    return image_paths
