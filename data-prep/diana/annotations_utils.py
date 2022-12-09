import os
import json
import time
import random
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, NamedTuple, Tuple, Union

from panorama.client import PanoramaClient

from visualizations.model import PointOfInterest
from visualizations.unique_instance_prediction import geo_clustering, generate_map, color_generator, append_geohash, \
    get_points
from dataclass_wizard import Container


def collect_pano_ids(srcs: List[str], exclude_prefix: str = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a list of json annotation files in COCO format, return image filenames.

    Args:
        srcs: list of paths to json files
        exclude_prefix: pattern filename to exclude

    return image filenames, ids of images, zipped!
    """
    filenames_as_list = []
    filenames_TMX_as_list = []
    locations_as_list = []
    ids_as_list = []
    ids_TMX_as_list = []
    for src in srcs:
        with open(src, "r") as read_file:
            content = json.load(read_file)
        read_file.close()

        images_obj = content["images"]
        for image_obj in images_obj:
            location = image_obj["file_name"].split("/")[-3]
            filename = os.path.splitext(os.path.basename(image_obj["file_name"]))[0]
            id_ = image_obj["id"]
            filenames_as_list.append(filename)
            ids_as_list.append(id_)
            locations_as_list.append(location)

    print(f"Nr images: {len(filenames_as_list)}")
    print(f"Nr UNIQUE images: {len(set(filenames_as_list))}.")

    assert len(filenames_as_list) == len(ids_as_list)

    if exclude_prefix:

        pano_filenames = [fn for fn in list(filenames_as_list) if fn.startswith(exclude_prefix)]
        print(f"Nr images which start with pano: {len(pano_filenames)}.")
        print(f"Nr UNIQUE images which start with pano: {len(set(pano_filenames))}.")

        for filename, id_ in zip(filenames_as_list, ids_as_list):
            if not filename.startswith(exclude_prefix):
                filenames_TMX_as_list.append(filename)
                ids_TMX_as_list.append(id_)

        print(f"Nr images which start with TMX  {len(filenames_TMX_as_list)}.")
        print(f"Nr UNIQUE images which start with TMX. {len(set(filenames_TMX_as_list))}.")

    assert len(filenames_TMX_as_list) == len(ids_TMX_as_list)
    assert len(ids_TMX_as_list) == len(set(ids_TMX_as_list)), "ids are not unique"

    return filenames_TMX_as_list, ids_TMX_as_list, locations_as_list


def get_filenames_metadata(filenames: List[str], ids: List[str], locations: List[str]) -> Container[PointOfInterest]:
    """

    return map with images_coords
    """
    filenames_loc = Container[PointOfInterest]()

    for filename, img_id, loc in tqdm(zip(filenames, ids, locations), total=len(filenames),
                                      desc="Getting metadata from API"):
        try:
            pano_object = PanoramaClient.get_panorama(filename)
            lat = pano_object.geometry.coordinates[1]
            lon = pano_object.geometry.coordinates[0]

            filenames_loc.append(
                PointOfInterest(
                    pano_id=filename,
                    coords=(lat, lon),
                    image_id=img_id,
                    location=loc
                )
            )
            # panorama API allows 6 requests per 10 seconds = 1 request per 0.6 seconds
            time.sleep(0.6)
        except ValueError:
            print(f"{filename} failed.")
    return filenames_loc


def split_pano_ids(points, nr_clusters, train_ratio=0.7, val_ratio=0.15):
    """
    Split panorama filenames in train.txt, val.txt and test.txt based on given split ratio
    """

    total = len(points)   # add the pano files

    print(f"total: {total}")
    train_count = int(train_ratio * total)

    val_count = int(val_ratio * total)

    train_points = []
    val_points = []
    test_points = []

    print(f"Number of panorama files: {len(points)}")
    print(f"Train count is {train_count}, val count is {val_count}.")

    def _not_full(a_list, threshold):
        if len(a_list) < threshold:
            return True
        return False

    # shuffle the cluster indices. If we don't we get a lot of test points in Nieuw West
    cluster_indices = list(np.arange(nr_clusters))
    random.seed(4)
    random.shuffle(cluster_indices)

    for cluster_id in cluster_indices:
        points_subset = get_points(points, cluster_id=cluster_id)
        if _not_full(train_points, threshold=train_count):
            for i, point in enumerate(points_subset):
                points_subset[i].subset = "train"
            train_points.extend(points_subset)
        elif _not_full(val_points, threshold=val_count):
            for i, point in enumerate(points_subset):
                points_subset[i].subset = "val"
            val_points.extend(points_subset)
        else:
            for i, point in enumerate(points_subset):
                points_subset[i].subset = "test"
            test_points.extend(points_subset)

    print(f"After filename splitting, train count is {len(train_points)}, val count is {len(val_points)}, "
          f"test count is {len(test_points)}.")

    return train_points, val_points, test_points

