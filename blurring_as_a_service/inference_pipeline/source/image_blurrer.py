import os

import cv2
from cv2.mat_wrapper import Mat


class ImageBlurrer:
    def __init__(self, image_path: str):
        self._image_to_blur = cv2.imread(image_path)

    def blur(self, labels_path: str) -> Mat:
        """
        Blur the image based on the labels inside the file in labels_path.
        Each line of the file is:
            label_type normalized_x_value normalized_y_value normalized_weight normalized_height

        Parameters
        ----------
        labels_path
            Path to the file containing the areas to be blurred.

        Returns
        -------

        Blurred image
        """
        with open(
            labels_path,
            "r",
        ) as labels_file:
            for label in labels_file:
                _, x_norm, y_norm, w_norm, h_norm = label.split()
                image = self._blur_image_region(
                    float(x_norm),
                    float(y_norm),
                    float(w_norm),
                    float(h_norm),
                )

        return image

    def blur_and_store(self, labels_path: str, store_path: str):
        """
        Blur the image based on the labels inside the file in labels_path and stores the result image in store_path.

        Parameters
        ----------
        labels_path
            Path to the file containing the areas to be blurred.
        store_path
            Path where to store the blurred image.

        """
        image = self.blur(labels_path=labels_path)
        folder_path = os.path.dirname(store_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not cv2.imwrite(
            store_path,
            image,
        ):
            raise Exception(f"Could not write image {os.path.basename(store_path)}")

    def _blur_image_region(
        self, x_norm: float, y_norm: float, w_norm: float, h_norm: float
    ) -> Mat:
        """
        Apply gaussian blur to a part of an image.

        Parameters
        ----------
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
        height, width = self._image_to_blur.shape[:2]
        x, y = int(x_norm * width), int(y_norm * height)
        w, h = int(w_norm * width), int(h_norm * height)
        x1, x2 = round(x - w / 2), round(x + w / 2)
        y1, y2 = round(y - h / 2), round(y + h / 2)

        # Get the region of interest from the image
        roi = self._image_to_blur[y1:y2, x1:x2]
        # Apply Gaussian blur to the region
        blur = cv2.GaussianBlur(roi, (135, 135), 0)
        # Replace the original region with the blurred region
        self._image_to_blur[y1:y2, x1:x2] = blur
        return self._image_to_blur
