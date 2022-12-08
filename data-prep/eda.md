  
### Azure COCO format   
We download the annotations from AzureML.  
The used annotation file can be found [here](diana/in:coco-format/blur_v0.1/blur-annotations.json).  
  
  
Let's look at the structure of the current annotation file  
  
It has a(n) (Azure version of ) COCO format.  
Id's are in range `[1, 1446]` and look like this:  
```python  
 "images": [  
 { "id": 1, "width": 8000.0, "height": 4000.0, "file_name": "annotations-projects/07-25-2022_120550_UTC/blurring-project/Westermarkt/images/TMX7316010203-001666_pano_0000_000927.jpg", "coco_url": "AmlDatastore://annotations_datastore/annotations-projects/07-25-2022_120550_UTC/blurring-project/Westermarkt/images/TMX7316010203-001666_pano_0000_000927.jpg", "absolute_url": "https://cvodataweupgwapeg4pyiw5e.blob.core.windows.net/annotations-blob-container/annotations-projects/07-25-2022_120550_UTC/blurring-project/Westermarkt/images/TMX7316010203-001666_pano_0000_000927.jpg", "date_captured": "2022-09-16T14:48:15.6141099Z" },```  
Images have a `8000x4000` resolution.  
  
Annotations are in range `[1, 12651]` and they look like this:  
```python  
"annotations": [  
 { "id": 1, "category_id": 2, "image_id": 1, "area": 0.0, "bbox": [ 0.004358247984310308, 0.5711726108326676, 0.013801118616982641, 0.006779496864482648 ] { "id": 5211, "category_id": 2, "image_id": 595, "area": 0.0, "bbox": [ 0.16222851746931066, 0.5514636449480642, 0.008404154863078384, 0.004343720491029268 ] }, { "id": 5212, "category_id": 2, "image_id": 595, "area": 0.0, "bbox": [ 0.22502360717658168, 0.5299338999055713, 0.0026440037771482405, 0.0026440037771482405 ]```  
It looks like the bbox is in the normalized `[x, y, width, height]` format.  
  
Categories look like this  
```python  
  "categories": [  
 { "id": 1, "name": "Person" }, { "id": 2, "name": "Licence plate" } ]```  
---  
### Yolov5 format   
  
To convert the Azure COCO dataset into a yolov5 format we have to do the following:  
  
- We have 1446 images, so we will end up with 1446 txt files.  
An example of one txt file:  
```text  
45 0.479492 0.688771 0.955609 0.5955  
45 0.736516 0.247188 0.498875 0.476417  
50 0.637063 0.732938 0.494125 0.510583  
45 0.339438 0.418896 0.678875 0.7815  
49 0.646836 0.132552 0.118047 0.0969375  
  
[class name] [x] [y] [width] [height]  
```  
- The bbox will remain the same format  
- Categories will change from 1=person and 2=license plate to 0=person and 1=licence plate  
- if we split train-val in for example 1000-446 we have `train/images` with 1000 jpg files  and `train/labels` with 1000 txt files  
and `val/images` with 446 jpg files and `val/labels` with 446 txt files.  
- we have a main `blur-dataset.yaml` which contains the description of the dataset.  
  
The `blur-dataset.yaml` will look like this  
  
```yaml  
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license  
# Blurring dataset   
# Example usage: python train.py --data-prep blur.yaml  
# parent  
# ├── Blurring-as-a-Service  
# └── datasets  
#     └── blur  
  
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]  
path: ../datasets/blur  # dataset root dir  
train: images/train  # train images (relative to 'path') 128 images  
val: images/val  # val images (relative to 'path') 128 images  
test:  # test images (optional)  
  
# Classes  
names:  
  0: person  
  1: license_plate  
  
# Download script/URL (optional)  
download: -  
```  
---  
### Data conversion  
  
```text  
for each image in coco format:  
    retrieve all its annotations  
    convert class name: if category=1 make it 0, if category=2 make it 1  
    store them in a txt file  
    save txt file with image name  
```  
```python  
from utils import CocoToYoloConverter  
  
source = "in:coco-format/blur_v0.1/blur-annotations.json"  
output_dir = "out:yolo-format/blur_v0.1"  
  
converter = CocoToYoloConverter(source, output_dir)  
converter.convert()  
```  
We generated 1445 txt files with annotations in out:yolo-format/blur_v0.1  
One is a duplicate!  
  
  
### Inspecting the old dataset  
  
The old dataset is already in YOLO format, and it has been used in the past for training a blurring model.  
The old dataset can be found at the following location in the [AS storage account](https://portal.azure.com/#view/Microsoft_Azure_Storage/ContainerMenuBlade/~/overview/storageAccountId/%2Fsubscriptions%2Fb5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14%2FresourceGroups%2Fcvo-aml-p-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fcvodataweupgwapeg4pyiw5e/path/annotations-blob-container/etag/%220x8DAD3B5819D0865%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride~/false/defaultId//publicAccessVal/None),   
at `annotations-projects/07-25-2022_120550_UTC/old-blurring-dataset/` where we have   
  
- `images` folder with 3744 blobs  
- `labels` folder with 3750 blobs  
  
For the blurring we want a map with all locations of all new and old images, that is a map with  
roughly 1446 + 3744 = 5190 images. For all these images we will compute some metadata in order to help us  
better split the dataset inclusively.  
  
We know the old dataset has 3744 images, but let's get their ids, store them in a file so we have easy access to them.  

```python  
import os  
import json  
from azureml.core import (  
    Dataset,  
    Environment,  
    Workspace,  
)  
  
ws = Workspace.from_config()  
env = Environment.from_dockerfile("blurring-env", "data-prep/laurens/blur-environment")  
dataset = Dataset.get_by_name(ws, "blur_v1")  
paths = dataset.to_path()  
  
paths_new_blur = [path for path in paths if path.startswith("/blurring-project/") and path.endswith(".jpg")]  
paths_old_blur = [path for path in paths if path.startswith("/old-blurring-dataset/images/") and path.endswith(".jpg")]  
  
  
pano_ids_old_blur = [os.path.splitext(os.path.basename(path))[0] for path in paths_old_blur]  
print(f"{len(paths_old_blur)} old jpg files from the previously trained blurred model.") 
  
with open('in:coco-format/blur_v0.1/old-blurring-ids.json', 'w') as outfile:  
    json.dump(pano_ids_old_blur, outfile)  
```
  
  The output is:
```
3742 old jpg files from the previously trained blurred model.
```
  
  
Let's now create the new list of PointOfInterest with location being None and update metadata_for_map.json.   
  
```python  
import json  
from annotations_utils import get_filenames_metadata  
from dataclass_wizard import Container  
from visualizations.unique_instance_prediction import append_geohash  
from visualizations.model import PointOfInterest  
  
with open("in:coco-format/blur_v0.1/old-blurring-ids.json", "r") as read_file:  
    old_file_names = json.load(read_file)  
read_file.close()  
  
ids = len(old_file_names)*["-1"]  
locations = len(old_file_names) * ["None"]  
# create Points of Interest list  
points = get_filenames_metadata(old_file_names, ids, locations)  
  
# add geohash to Points of Interest list  
new_points_geohash = append_geohash(points)  
  
# append data to json  
old_list = PointOfInterest.from_json_file('in:coco-format/blur_v0.1/metadata_for_map.json')  
updated_list = old_list + new_points_geohash  
print(f"Initial json file had {len(old_list)} points of interest.")  
Container[PointOfInterest](updated_list).to_json_file("in:coco-format/blur_v0.1/metadata_for_map.json", indent=4)  
  
check_updated_list = PointOfInterest.from_json_file('in:coco-format/blur_v0.1/metadata_for_map.json')  
print(f"After adding {len(new_points_geohash)} points of interest, we have {len(check_updated_list)} points of interest.")  
```  
The output is:   
```text   
Initial json file had 1446 points of interest.
After adding 3742 points of interest, we have 5188 points of interest. 
```  

## Creating the ORBS-base dataset

ORBS-base dataset = old dataset + 1st batch of 1445 images annotated by Liska. 

Let's call the 1st batch of 1445 images annotated by Liska `ORBS-first-batch` dataset

We need images from  `ORBS-first-batch` in a single folder, instead of a dozen of subfolders with location names.
For this, we take `file_names` corresponding to `ORBS-first-batch` images from the `metadata_for_map.json`  and we download all these images.
Download source:  `blurring-project` blob 
Target path: `in:coco-format/blur_v0.1/all/images`
  
```python  
from visualizations.model import PointOfInterest  
from azure.identity import DefaultAzureCredential  
from azure.storage.blob import ContainerClient  
  
STORAGE_ACCOUNT = 'https://cvodataweupgwapeg4pyiw5e.blob.core.windows.net/'  
CONTAINER_NAME = 'annotations-blob-container'  
  
  
points = PointOfInterest.from_json_file('in:coco-format/blur_v0.1/metadata_for_map.json')  
panorama_ids = [point.pano_id for point in points if point.location != "None"]  
    
def download_images_from_datastore(pano_ids):  
    default_credential = DefaultAzureCredential()  
    files_found = 0  
    # Create the BlobServiceClient object  
    blob_service_client = ContainerClient(STORAGE_ACCOUNT, container_name=CONTAINER_NAME, credential=default_credential)  
    for file in blob_service_client.list_blobs():  
        if file.name.endswith(".jpg"):  
            filename = file.name.split("/")[-1].split(".")[0]  
            if filename in pano_ids:  
                files_found += 1  
                # download image  
                with open(file=f"in:coco-format/blur_v0.1/all/images{filename}.jpg", mode="wb") as download_file:  
                    download_file.write(blob_service_client.download_blob(file.name).readall())  
    print(f"{files_found} downloaded files.")  
  
  
download_images_from_datastore(panorama_ids)  
  
```  
  
We downloaded the 1445 images (we have one duplicate).   
Previously, we generated the new annotations txt files.  

---
Now we create the `ORBS-base` dataset at the following location in the storage account:
link

Here we have 2 folders: 

-   `images`  folder with ~5200 blobs
-   `labels`  folder with ~5200 blobs

!! *This is the dataset used from now onwards to train the first version of the blurring model.*
  
### Visualizations  
  
Let's look at the distribution of our images thoughout the city  
  
First, we query panorama API and store metadata of the images in json  
(so we don't send too many requests)  
```python  
  
from annotations_utils import collect_pano_ids  
from dataclass_wizard import Container  
from visualizations.unique_instance_prediction import append_geohash  
from visualizations.model import PointOfInterest  
from annotations_utils import get_filenames_metadata  
  
srcs = ["in:coco-format/blur_v0.1/blur-annotations.json"]  
  
# merge names of filenames from all annotation files  
file_names, ids, locations = collect_pano_ids(srcs, exclude_prefix="pano")  
  
# create Points of Interest list  
points = get_filenames_metadata(file_names, ids, locations)  
  
# add geohash to Points of Interest list  
points_geohash = append_geohash(points)  
  
# save data-prep to json  
# a `Container` object is just a wrapper around a Python `list`  
Container[PointOfInterest](points_geohash).to_json_file("in:coco-format/blur_v0.1/metadata_for_map.json", indent=4)  
```  
We can then generate a simple map   
  
```python  
from visualizations.unique_instance_prediction import generate_map  
from visualizations.model import PointOfInterest  
  
  
points_geohash = PointOfInterest.from_json_file('in:coco-format/blur_v0.1/metadata_for_map.json')  
  
generate_map(vulnerable_bridges=[],  
  permit_locations=[],  
  detections=points_geohash,  
  name="in:coco-format/blur_v0.1/all_locations")  
```  

BELOW IS OBSOLETE 
As stated above, let's see if the panoramas ids overlap. If they do, then we have   
multiple txt files for the same panorama id, which we do not want.  
```python  
import json  
from annotations_utils import collect_pano_ids  
  
srcs = ["in:coco-format/blur_v0.1/blur-annotations.json"]  
new_file_names, _, _ = collect_pano_ids(srcs, exclude_prefix="pano")  
  
with open("in:coco-format/blur_v0.1/old-blurring-ids.json", "r") as read_file:  
    old_file_names = json.load(read_file)  
read_file.close()  
  
all_file_names = new_file_names + old_file_names  
print(f"There is a total of {len(all_file_names)} images, {len(new_file_names)} new and {len(old_file_names)} old.")  
print(f"There is a total of {len(set(all_file_names))} unique images.")  
```  


  
