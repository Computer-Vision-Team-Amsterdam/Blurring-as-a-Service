from dataclasses import dataclass
from os import listdir
from os.path import isfile, join
from typing import List, Tuple

import geohash
from dataclass_wizard import Container, JSONFileWizard, JSONListWizard
from panorama.client import PanoramaClient


@dataclass
class Metadata(JSONListWizard, JSONFileWizard):
    pano_id: str
    coords: Tuple[float, float] = (-1, -1)
    geohash: str = ""
    location: str = ""
    cluster: int = 0
    subset: str = ""


class MetadataRetriever:
    def __init__(self, images_directory_path: str):
        self.images_directory_path = images_directory_path

    def generate_and_store_metadata(self, json_output_path_and_file_name: str):
        metadata_objects_to_store = self.generate_metadata()
        Container[Metadata](metadata_objects_to_store).to_json_file(
            json_output_path_and_file_name, indent=4
        )

    def generate_metadata(self) -> List[Metadata]:
        filenames = self._get_all_filenames_in_dir(self.images_directory_path)
        valid_filenames = self._filter_only_images_allowed_by_the_api(filenames)
        metadata_objects = [
            self._get_panorama_api_information(filename) for filename in valid_filenames
        ]
        return [self._append_geohash(filename) for filename in metadata_objects]

    @staticmethod
    def _get_all_filenames_in_dir(directory_path: str) -> List[str]:
        return [f for f in listdir(directory_path) if isfile(join(directory_path, f))]

    @staticmethod
    def _filter_only_images_allowed_by_the_api(
        images_available: List[str],
    ) -> List[str]:
        return [
            image_name
            for image_name in images_available
            if image_name.startswith("TMX")
        ]

    @staticmethod
    def _get_panorama_api_information(image_name: str) -> Metadata:
        image_name = (
            image_name[: -len(".jpg")] if image_name.endswith(".jpg") else image_name
        )
        panorama_object = PanoramaClient.get_panorama(image_name)

        latitude = panorama_object.geometry.coordinates[1]
        longitude = panorama_object.geometry.coordinates[0]
        return Metadata(
            pano_id=image_name,
            coords=(latitude, longitude),
        )

    @staticmethod
    def _append_geohash(
        metadata_to_enhance: Metadata,
    ) -> Metadata:
        metadata_to_enhance.geohash = geohash.encode(
            metadata_to_enhance.coords[0], metadata_to_enhance.coords[1]
        )
        return metadata_to_enhance
