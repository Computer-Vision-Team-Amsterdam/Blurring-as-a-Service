"""
This module is responsible for visualizing the trajectory and container found on a day.
Show the containers that were found on the particular trajectory that was driven for a particular day.

Pano API sanity check:

from panorama import models
from panorama.client import PanoramaClient

# Get the first page of panoramas
response: models.PagedPanoramasResponse = PanoramaClient.list_panoramas()
"""
import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Union

from panorama import models  # pylint: disable=import-error
from panorama.client import PanoramaClient  # pylint: disable=import-error
from triangulation.helpers import (
    get_panos_from_points_of_interest,
)  # pylint: disable-all

from visualizations.model import PointOfInterest
from visualizations.unique_instance_prediction import generate_map
from visualizations.utils import get_bridge_information, get_permit_locations


def get_daily_panoramas(
    target_date: date, location_query: models.LocationQuery
) -> models.PagedPanoramasResponse:
    """
    This method queries the panorama API for all panorama objects stored at a specific date.

    :param target_date: date we are interested to know trajectory for
    :param location_query: search query
    :returns: paged list of panorama objects based on query

    """
    query_result: models.PagedPanoramasResponse = PanoramaClient.list_panoramas(
        location=location_query,
        timestamp_after=target_date,
        timestamp_before=target_date + timedelta(days=1),
    )
    return query_result


def get_panorama_coords(
    daily_panoramas: models.PagedPanoramasResponse,
) -> List[List[float]]:
    """
    This method collects the coordinates of the panorama objects stored at a specific date
    such that their timestamps are in chronological order

    :returns: list of lists of [latitude, longitude]
    """
    if len(daily_panoramas.panoramas) == 0:
        raise ValueError("No available panoramas.")

    total_pano_pages = int(daily_panoramas.count / 25)
    print(f"There is a total of {total_pano_pages} panorama pages to iterate over.")
    print(50 * "=")
    pano_page_count = 0
    scan_coords = []
    timestamps = []
    while True:
        pano_page_count = pano_page_count + 1
        if pano_page_count % 20 == 0:
            print(f"Finished {pano_page_count} out of {total_pano_pages}.")
        try:
            for i in range(len(daily_panoramas.panoramas)):
                panorama: models.Panorama = daily_panoramas.panoramas[i]
                long, lat, _ = panorama.geometry.coordinates
                timestamps.append(panorama.timestamp)
                scan_coords.append([lat, long])

            next_pano_batch: models.PagedPanoramasResponse = PanoramaClient.next_page(
                daily_panoramas
            )
            daily_panoramas = next_pano_batch
        except ValueError:
            print("No next page available")
            break

    sorted_lists = sorted(zip(timestamps, scan_coords), key=lambda x: x[0])  # type: ignore
    sorted_timestamps, sorted_coords = [[x[i] for x in sorted_lists] for i in range(2)]

    return sorted_coords


def run(
    day_to_plot: date,
    location_query: models.LocationQuery,
    points_of_interest: Union[Path, str],
    vulnerable_bridges_file: Union[Path, str],
    permits_file: Union[Path, str],
) -> None:
    """
    This method creates visualization of a path and detected containers based on trajectory on a specific date.

    :param day_to_plot: target date.
    :param location_query: location information for API search
    :param points_of_interest: path to triangulation output file.
    :param vulnerable_bridges_file: path to vulnerable bridges input file.
    :param permits_file: path to decos permits input file.
    """

    # ========= CREATE CAR TRAJECTORY =================

    daily_query_result = get_daily_panoramas(day_to_plot, location_query)

    trajectory = get_panorama_coords(daily_query_result)  # only keep their coordinates

    # ======== CREATE LIST OF DETECTIONS ============

    detections = []
    panoramas = get_panos_from_points_of_interest(
        points_of_interest,
        timestamp_after=day_to_plot,
        timestamp_before=day_to_plot + timedelta(days=1),
    )

    with open(points_of_interest, "r") as file:
        reader = csv.reader(file)
        next(reader)  # skip first line
        for i, row in enumerate(reader):
            detections.append(
                PointOfInterest(
                    pano_id=panoramas[i].id, coords=(float(row[0]), float(row[1]))
                )
            )

    # ======== CREATE LIST OF VULNERABLE BRIDGES ============
    vulnerable_bridges = get_bridge_information(vulnerable_bridges_file)

    # ======== CREATE LIST OF PERMIT LOCATIONS ============
    date_to_check = datetime(2021, 3, 17)
    permit_locations = get_permit_locations(permits_file, date_to_check)

    # ========== CREATE MAP =================
    generate_map(
        vulnerable_bridges,
        permit_locations,
        trajectory=trajectory,
        detections=detections,
    )


if __name__ == "__main__":

    target_day = date(2021, 3, 17)

    # Kloveniersburgwal 45
    lat = 52.370670
    long = 4.898990
    radius = 2000
    location_query = models.LocationQuery(latitude=lat, longitude=long, radius=radius)

    coordinates = "points_of_interest.csv"
    vulnerable_bridges_file = "bridges.geojson"
    permits_file = "decos.xml"
    run(target_day, location_query, coordinates, vulnerable_bridges_file, permits_file)
