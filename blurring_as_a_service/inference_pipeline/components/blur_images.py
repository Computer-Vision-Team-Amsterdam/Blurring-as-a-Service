import os
import sys

import cv2
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


@command_component(
    name="blur_images",
    display_name="Blur images.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
)
def blur_images(
    data_to_blur: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_detection: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    results_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    images_to_blur = os.listdir(data_to_blur)
    detection_result_files = os.listdir(f"{results_detection}/detection_result/labels")
    detection_result_files = [
        os.path.splitext(os.path.basename(path))[0] for path in detection_result_files
    ]
    for image_to_blur in images_to_blur:
        images_to_blur_filename = os.path.splitext(image_to_blur)[0]
        if images_to_blur_filename in detection_result_files:
            image = cv2.imread(f"{data_to_blur}/{image_to_blur}")
            with open(
                f"{results_detection}/detection_result/labels/{images_to_blur_filename}.txt",
                "r",
            ) as labels_file:
                for label in labels_file:
                    label_coordinates = label.split()
                    _, x_norm, y_norm, w_norm, h_norm = label_coordinates
                    image = blur_image_region(image, x_norm, y_norm, w_norm, h_norm)
            print(
                f"Writing blurred image in {results_path}/detection_result/{os.path.basename(image_to_blur)}"
            )
            if not cv2.imwrite(
                f"{results_path}/detection_result/{os.path.basename(image_to_blur)}",
                image,
            ):
                raise Exception("Could not write image")
        else:
            # TODO: Move the image anyway
            print(f"Not to blur {image_to_blur}")


def blur_image_region(image, x_norm, y_norm, w_norm, h_norm):
    # Convert the normalized coordinates to pixel coordinates
    height, width = image.shape[:2]
    x, y = int(float(x_norm) * width), int(float(y_norm) * height)
    w, h = int(float(w_norm) * width), int(float(h_norm) * height)
    x1, x2 = round(x - w / 2), round(x + w / 2)
    y1, y2 = round(y - h / 2), round(y + h / 2)

    # Get the region of interest from the image
    roi = image[y1:y2, x1:x2]
    # Apply Gaussian blur to the region
    blur = cv2.GaussianBlur(roi, (135, 135), 0)
    # Replace the original region with the blurred region
    image[y1:y2, x1:x2] = blur

    return image
