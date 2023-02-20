import os
import shutil
import sys

import cv2
from azure.ai.ml.constants import AssetTypes
from cv2.mat_wrapper import Mat
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
    """
    Pipeline step to blur the images. The areas to blur have already been identified by the model in a previous step.

    Parameters
    ----------
    data_to_blur:
        Images to be blurred
    results_detection:
        Areas to blur determined by the model.
    results_path:
        Where to store the results.
    """
    detection_result_files = os.listdir(f"{results_detection}/detection_result/labels")
    detection_result_files = [
        os.path.splitext(os.path.basename(path))[0] for path in detection_result_files
    ]
    for image_to_blur in os.listdir(data_to_blur):
        images_to_blur_filename = os.path.splitext(image_to_blur)[0]
        if images_to_blur_filename in detection_result_files:
            image = cv2.imread(f"{data_to_blur}/{image_to_blur}")
            with open(
                f"{results_detection}/detection_result/labels/{images_to_blur_filename}.txt",
                "r",
            ) as labels_file:
                for label in labels_file:
                    _, x_norm, y_norm, w_norm, h_norm = label.split()
                    image = blur_image_region(
                        image,
                        float(x_norm),
                        float(y_norm),
                        float(w_norm),
                        float(h_norm),
                    )
            if not cv2.imwrite(
                f"{results_path}/detection_result/{os.path.basename(image_to_blur)}",
                image,
            ):
                raise Exception(
                    f"Could not write image {os.path.basename(image_to_blur)}"
                )
        else:
            shutil.copyfile(
                f"{data_to_blur}/{image_to_blur}",
                f"{results_path}/detection_result/{os.path.basename(image_to_blur)}",
            )


def blur_image_region(
    image: Mat, x_norm: float, y_norm: float, w_norm: float, h_norm: float
) -> Mat:
    """
    Apply gaussian blur to a part of an image.

    Parameters
    ----------
    image:
        Image to be blurred
    x_norm:
        X value of the part to blur normalized
    y_norm:
        Y value of the part to blur normalized
    w_norm:
        Weight value of the part to blur normalized
    h_norm:
        Height value of the part to blur normalized

    Returns
    -------
        original image with the specified part blurred

    """
    # Convert the normalized coordinates to pixel coordinates
    height, width = image.shape[:2]
    x, y = int(x_norm * width), int(y_norm * height)
    w, h = int(w_norm * width), int(h_norm * height)
    x1, x2 = round(x - w / 2), round(x + w / 2)
    y1, y2 = round(y - h / 2), round(y + h / 2)

    # Get the region of interest from the image
    roi = image[y1:y2, x1:x2]
    # Apply Gaussian blur to the region
    blur = cv2.GaussianBlur(roi, (135, 135), 0)
    # Replace the original region with the blurred region
    image[y1:y2, x1:x2] = blur

    return image
