import unittest

import numpy as np
from numpy.testing import assert_allclose

from blurring_as_a_service.metrics.custom_metrics_calculator import (
    ImageSize,
    TargetClass,
)
from blurring_as_a_service.utils.yolo_labels_dataset import YoloLabelsDataset


def expected_output_get_item():
    labels = np.array(
        [
            [
                0,
                0.2788512911843277,
                0.5150489759572574,
                0.0034728406055209438,
                0.024220837043633092,
            ],
            [
                1,
                0.32333036509349955,
                0.5315227070347284,
                0.0017809439002671734,
                0.00267141585040076,
            ],
            [
                1,
                0.3346393588601959,
                0.536420302760463,
                0.0014247551202137276,
                0.003918076580587737,
            ],
            [
                0,
                0.5972374660895856,
                0.5734225185860135,
                0.022365342003354827,
                0.15200149102280025,
            ],
            [
                0,
                0.6213630433432045,
                0.5527138685829072,
                0.013046449501956991,
                0.089047195013357,
            ],
            [
                1,
                0.6924309884238646,
                0.5253784505788068,
                0.006233303650934996,
                0.003918076580587626,
            ],
            [
                0,
                0.7214158504007124,
                0.5158504007123775,
                0.006500445235975039,
                0.027248441674087243,
            ],
            [
                0,
                0.7340605520926091,
                0.5219946571682992,
                0.0075690115761353205,
                0.0356188780053428,
            ],
            [
                0,
                0.9220967995390499,
                0.5464616835157928,
                0.012178513435650351,
                0.09926143208841864,
            ],
            [
                0,
                0.9401026661777802,
                0.5658425436069352,
                0.020690377664868365,
                0.08564244932166987,
            ],
        ]
    )
    return labels


class TestYoloLabelsDataset(unittest.TestCase):
    def setUp(self):
        self.folder_path = "../../local_test_data/sample/labels/val"
        self.dataset = YoloLabelsDataset(self.folder_path)

    def test_dataset_initialization(self):
        self.assertEqual(self.dataset.folder_path, self.folder_path)
        self.assertGreater(len(self.dataset.label_files), 0)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), len(self.dataset.label_files))
        self.assertEqual(len(self.dataset.get_labels()), 4)

    def test_dataset_getitem(self):
        image_id = "TMX7316010203-001041_pano_0002_002709"
        assert_allclose(self.dataset[image_id], expected_output_get_item())

    def test_filter_by_class(self):
        self.dataset.filter_by_class(class_to_keep=TargetClass.person.value)
        labels_class_person = self.dataset.get_filtered_labels()
        for image_id, labels in labels_class_person.items():
            assert np.all(
                labels[:, 0] == TargetClass.person.value
            ), "First column should contain only 0s"

    def test_filter_by_size(self):
        self.dataset.filter_by_size(
            size_to_keep=ImageSize.small.value, image_area=32000000
        )
        labels_size_small = self.dataset.get_filtered_labels()
        min_size = ImageSize.small.value[0]
        max_size = ImageSize.small.value[1]
        for image_id, labels in labels_size_small.items():
            assert np.logical_and(
                labels[:, 2] * labels[:, 3] >= min_size,
                labels[:, 2] * labels[:, 3] < max_size,
            ).all()
