import logging
import os
import shutil

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


def delete_file(file_path):
    try:
        os.remove(file_path)
        logger.info(f"{file_path} has been deleted.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to remove file '{file_path}': {str(e)}")
        raise Exception(f"Failed to remove file '{file_path}': {e}")


def copy_file(relative_path, input_path, output_path):
    source_path = os.path.join(input_path, relative_path)
    destination_path = os.path.join(output_path, relative_path)

    logger.info(f"Copying {source_path} to {destination_path}..")
    if os.path.exists(source_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(source_path, destination_path)
        print(f"File '{source_path}' copied to '{destination_path}'")

        if not os.path.exists(destination_path):
            error_message = f"Failed to move file '{source_path}' to the destination: {destination_path}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)
    else:
        logger.info(f"Source file '{source_path}' does not exist.")
