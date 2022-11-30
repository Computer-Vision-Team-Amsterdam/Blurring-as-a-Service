from azureml.core import Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import shutil
from datetime import datetime
from typing import Tuple
from collections import namedtuple
import csv
from pathlib import Path
import folium
from datetime import date

from panorama.client import PanoramaClient
from panorama import models
import requests
from requests.auth import HTTPBasicAuth

WORKSPACE = Workspace.from_config()
BASE_URL = f"https://3206eec333a04cc980799f75a593505a.objectstore.eu/intermediate/"

Location = namedtuple('Location', ['address', 'lon', 'lat', 'radius', 'start_year', 'end_year'])


def split_pano_id(pano_id: str) -> Tuple[str, str]:
    """
    Splits name of the panorama in TMX* and pano*
    """
    id_name = pano_id.split("_")[0]
    index = pano_id.index("_")
    img_name = pano_id[index + 1 :]
    return id_name, img_name


def download_panorama_from_cloudvps(
    date: datetime, panorama_id: str, output_dir: Path = Path("./retrieved_images")
) -> None:
    """
    Downloads panorama from cloudvps to local folder.
    """
    keyvault = WORKSPACE.get_default_keyvault()
    username_cloudvps = keyvault.get_secret(name="cloudvps-username")
    password_cloudvps = keyvault.get_secret(name="clousvps-password")
    if Path(f"./{output_dir}/{panorama_id}.jpg").exists():
        # print(f"Panorama {panorama_id} is already downloaded.")
        return
    id_name, img_name = split_pano_id(panorama_id)

    try:
        url = (
            BASE_URL
            + f"{date.year}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/{id_name}/{img_name}.jpg"
        )

        response = requests.get(
            url, stream=True, auth=HTTPBasicAuth(username_cloudvps, password_cloudvps)
        )
        if response.status_code == 404:
            raise FileNotFoundError(f"No resource found at {url}")
        filename = f"./{output_dir}/{panorama_id}.jpg"
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
    except Exception as e:
        print(f"Failed for panorama {panorama_id}:\n{e}")


def read_locations_file(location_file):
    locations = []
    with open(location_file) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for row in csv_reader:
            locations.append(
                Location(row['location'], float(row['long']), float(row['lat']), float(row['radius']), int(row['start jaar']), int(row['eind jaar'])))
    return locations


def get_pano_ids_from_location(location):
    pano_ids = []
    timestamp_after = date(location.start_year, 1, 1)
    timestamp_before = date(location.end_year, 1, 1)
    location_query = models.LocationQuery(
        latitude=location.lat,
        longitude=location.lon,
        radius=location.radius
    )
    query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
        location=location_query,
        timestamp_after=timestamp_after,
        timestamp_before=timestamp_before
    )
    while True:
        for pano in query_result.panoramas:
            pano_ids.append(pano)
        if len(pano_ids) == query_result.count:
            break
        query_result = PanoramaClient.next_page(query_result)
    return pano_ids


def sample_panos(panorama_ids, frequency):
    filtered_list = []
    for idx, pano in enumerate(sorted(panorama_ids, key=lambda x: x.id)):
        if idx % frequency == 0:
            filtered_list.append(pano)
    return filtered_list


def write_pano_list(path, panorama_ids):
    with open(path / "pano_ids.txt", 'w') as file:
        for pano in panorama_ids:
            file.write(pano.id + '\n')


def download_images(panorama_ids, path, location):
    image_folder = (path / location.address) / "images"
    image_folder.mkdir(exist_ok=True, parents=True)
    for panorama in panorama_ids:
        download_panorama_from_cloudvps(panorama.timestamp, panorama.id, image_folder)
    write_pano_list(image_folder.parent, panorama_ids)


def visualize_dataset(path, panorama_ids, locations):
    colors = {2016: 'red', 2017: "blue", 2018: 'green', 2019: "yellow", 2020: 'purple', 2021: 'black', 2022: 'orange'}
    map = folium.Map(location=[52.377956, 4.897070])
    for pano in panorama_ids:
        folium.Circle(
            radius=1,
            location=pano.geometry.coordinates[:2][::-1],
            popup=pano.id,
            color=colors[pano.timestamp.year]
        ).add_to(map)

    for location in locations:
        folium.Circle(
            radius=location.radius,
            location=[location.lat, location.lon],
            popup=location.address,
            color='brown'
        ).add_to(map)
    map.save(path / "index.html")


def upload_to_storage():
    datastore = Datastore.get(WORKSPACE, datastore_name="annotations_datastore")
    dataset = Dataset.get_by_name(WORKSPACE, name='ann-blurring-dataset')
    Dataset.File.upload_directory(src_dir='retrieved_images',
                                  target=DataPath(datastore,
                                                  "annotations-projects/07-25-2022_120550_UTC/blurring-project")
                                  )
    dataset.update()


def create_dataset_from_locations(file_to_locations="data-prep/locations.csv", sample_frequency=5, dataset_folder="Retrieved_images"):
    locations = read_locations_file(file_to_locations)
    filtered_pano_ids = []
    for location in locations:
        pano_ids = get_pano_ids_from_location(location)
        filtered_pano_id_per_location = sample_panos(pano_ids, sample_frequency)
        filtered_pano_ids.extend(filtered_pano_id_per_location)
        download_images(filtered_pano_id_per_location, dataset_folder, location)
    visualize_dataset(dataset_folder, filtered_pano_ids, locations)
    upload_to_storage()

    