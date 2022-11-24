
### Azure COCO format 
We download the annotations from AzureML.
The used annotation file can be found [here](prep-diana/coco-format/blur_v0.1/blur-annotations.json).


Let's look at the structure of the current annotation file

It has a(n) (Azure version of ) COCO format.
Id's are in range `[1, 1446]` and look like this:
```python
 "images": [
    {
      "id": 1,
      "width": 8000.0,
      "height": 4000.0,
      "file_name": "annotations-projects/07-25-2022_120550_UTC/blurring-project/Westermarkt/images/TMX7316010203-001666_pano_0000_000927.jpg",
      "coco_url": "AmlDatastore://annotations_datastore/annotations-projects/07-25-2022_120550_UTC/blurring-project/Westermarkt/images/TMX7316010203-001666_pano_0000_000927.jpg",
      "absolute_url": "https://cvodataweupgwapeg4pyiw5e.blob.core.windows.net/annotations-blob-container/annotations-projects/07-25-2022_120550_UTC/blurring-project/Westermarkt/images/TMX7316010203-001666_pano_0000_000927.jpg",
      "date_captured": "2022-09-16T14:48:15.6141099Z"
    },
```
Images have a `8000x4000` resolution.

Annotations are in range `[1, 12651]` and they look like this:
```python
"annotations": [
    {
      "id": 1,
      "category_id": 2,
      "image_id": 1,
      "area": 0.0,
      "bbox": [
        0.004358247984310308,
        0.5711726108326676,
        0.013801118616982641,
        0.006779496864482648
      ]
   {
      "id": 5211,
      "category_id": 2,
      "image_id": 595,
      "area": 0.0,
      "bbox": [
        0.16222851746931066,
        0.5514636449480642,
        0.008404154863078384,
        0.004343720491029268
      ]
    },
    {
      "id": 5212,
      "category_id": 2,
      "image_id": 595,
      "area": 0.0,
      "bbox": [
        0.22502360717658168,
        0.5299338999055713,
        0.0026440037771482405,
        0.0026440037771482405
      ]
```
It looks like the bbox is in the normalized `[x, y, width, height]` format.

Categories look like this
```python
  "categories": [
    {
      "id": 1,
      "name": "Person"
    },
    {
      "id": 2,
      "name": "Licence plate"
    }
  ]
```
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
- if we split train-val in 1000-446 we have `train/images` with 1000 jpg files  and `train/labels` with 1000 txt files
and `val/images` with 446 jpg files and `val/labels` with 446 txt files.
- we have a main `blur-dataset.yaml` which contains the description of the dataset.

The `blur-dataset.yaml` will look like this

```yaml
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Blurring dataset 
# Example usage: python train.py --data blur.yaml
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

source = "coco-format/blur_v0.1/blur-annotations.json"
output_dir = "yolo-format/blur_v0.1"

converter = CocoToYoloConverter(source, output_dir)
converter.convert()
```