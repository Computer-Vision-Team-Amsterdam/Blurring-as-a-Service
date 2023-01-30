import numpy as np

from PIL import Image
from unittest import TestCase

from metrics.utils import generate_mask, visualize_mask, parse_labels, generate_mask_binary, \
    visualize_mask_with_prediction


class Test(TestCase):

    def test_parse_labels(self):
        test_input = "../test-labels/val/TMX7316010203-000992_pano_0001_000323.txt"
        # Call the parse_file function
        first_elements, tuple_elements = parse_labels(test_input)
        # Check if the output is as expected
        expected_first_elements = [0, 0, 0, 0, 0, 0]
        assert first_elements == expected_first_elements, f"Expected {expected_first_elements} but got {first_elements}"
        expected_tuple_elements = [
            (0.6237570155750621, 0.5660134381788937, 0.02723971280184667, 0.11326378807027071),
            (0.6031440235076742, 0.5605816356638536, 0.020922329336564238, 0.11149882760277408),
            (0.9514871230099928, 0.5268506952785553, 0.010931820792947855, 0.04487870976249164),
            (0.11343916724709904, 0.5053653819038268, 0.007034648548473008, 0.024690352659174408),
            (0.07116150983514632, 0.5125950695996616, 0.009694145325033418, 0.02871124757164939),
            (0.06089907151492521, 0.5134588639366435, 0.008875015585180272, 0.030715615991602607),
        ]
        assert tuple_elements == expected_tuple_elements, f"Expected {expected_tuple_elements} but got {tuple_elements}"

    def test_generate_mask(self):
        # Test image
        image = Image.open("../test-images/val/TMX7316010203-000992_pano_0001_000323.jpg")
        # Test bounding boxes (normalized)
        bounding_boxes = [(0.2, 0.3, 0.1, 0.2), (0.5, 0.5, 0.1, 0.1)]
        # Get mask
        mask = generate_mask(bounding_boxes, image)
        # Test if mask has the same shape as image
        assert mask.size == image.size, f"Expected image size {image.size} but got {mask.size}"
        # Test if mask only contains pixels within bounding boxes
        mask_np = np.array(mask)
        image_np = np.array(image)
        for bounding_box in bounding_boxes:
            x_min = int(bounding_box[0] * image.width - bounding_box[2] * image.width / 2)
            y_min = int(bounding_box[1] * image.height - bounding_box[3] * image.height / 2)
            x_max = int(bounding_box[0] * image.width + bounding_box[2] * image.width / 2)
            y_max = int(bounding_box[1] * image.height + bounding_box[3] * image.height / 2)
            assert (mask_np[y_min:y_max, x_min:x_max, :] == image_np[y_min:y_max, x_min:x_max,
                                                            :]).all(), f"The mask does not contain only pixels within " \
                                                                       f"bounding box {bounding_box} "

    def test_generate_mask_binary(self):
        image = Image.open("../test-images/val/TMX7316010203-000992_pano_0001_000323-small.jpg")
        bounding_boxes = [(0.1, 0.1, 0.1, 0.1), (0.2, 0.2, 0.2, 0.2)]
        mask_binary = generate_mask_binary(bounding_boxes, image)
        assert isinstance(mask_binary, np.ndarray), f"Expected numpy array but got {type(mask_binary)}"
        assert mask_binary.shape == (4000, 8000), f"Expected shape (4000, 8000) but got {mask_binary.shape}"

    def test_visualize_mask(self):
        # Test image
        image = Image.open("../test-images/val/TMX7316010203-000992_pano_0001_000323-small.jpg")
        # Test bounding box
        classes, bounding_boxes = parse_labels("../test-labels/val/TMX7316010203-000992_pano_0001_000323.txt")
        # Get mask
        mask = generate_mask(bounding_boxes, image)
        # Test if visualize_mask() does not raise any exceptions
        try:
            visualize_mask(mask, image)
        except Exception as e:
            assert False, f"visualize_mask() raised an exception: {e}"

