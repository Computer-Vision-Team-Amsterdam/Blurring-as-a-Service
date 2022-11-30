from dataclasses import dataclass
from typing import Tuple
from dataclass_wizard import JSONListWizard, JSONFileWizard


@dataclass
class PointOfInterest(JSONListWizard, JSONFileWizard):
    pano_id: str
    coords: Tuple[float, float] = (-1, -1)
    geohash: str = ""
    location: str = ""
    cluster: int = 0
    subset: str = ""
    image_id: int = -1
