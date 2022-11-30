# Blurring-as-a-Service

This project is about removing personal data, i.e. persons and licence plates from raw panorama images.
We aim to do this in an inclusive manner, and we select our images based on different types of biases which can occur.
We create a document with a list of potential biases that we want to minimize and we select 
the panorama images used for training the model based on this document.

### Useful links
1. [Panorama API](https://api.data.amsterdam.nl/panorama/panoramas)
2. [Panorama viewer](https://data.amsterdam.nl/data/geozoek/?modus=kaart&term=Panoramabeelden&lagen=pano-pano2022bi%7Cpano-pano2021bi%7Cpano-pano2020bi%7Cpano-pano2019bi%7Cpano-pano2018bi%7Cpano-pano2017bi%7Cpano-pano2016bi%7Cpano-pano2021woz%7Cpano-pano2020woz%7Cpano-pano2019woz%7Cpano-pano2018woz%7Cpano-pano2017woz&legenda=true)
3. [Excel sheet with inclusivity biases (risico's en maatregelen.xlsx)](https://hoofdstad.sharepoint.com/sites/DigitaliseringenCTO/Shared%20Documents/Forms/AllItems.aspx?RootFolder=%2Fsites%2FDigitaliseringenCTO%2FShared%20Documents%2FInnovatie%20en%20RenD%2FComputer%20Vision%20Team%2FProjecten%2FInnovatiebudget%20%28hieronder%20valt%20Blur%20use%20case%29%2FInclusiviteit&FolderCTID=0x0120002EC45AFB501BC64FB525D14106AF3E05)
4. [Annotation project in Azure ML](https://ml.azure.com/labeling/project/93e9b2be-62de-6a8c-9c22-5b20cc5b90af/details?wsid=/subscriptions/b5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14/resourceGroups/cvo-aml-p-rg/providers/Microsoft.MachineLearningServices/workspaces/cvo-weu-aml-p-xnjyjutinwfyu&tid=72fca1b1-2c2e-4376-a445-294d80196804)
---

### Train custom dataset with yolov5
Source can be found [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data). 

Expected format:
- one *.txt file per image (if no objects in image, no *.txt file is required). 

The *.txt file specifications are:
- One row per object
- Each row is class x_center y_center width height format.
- Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
- Class numbers are zero-indexed (start from 0).
---

### Data processing 
More information about the exploratory data analysis can be found in the [eda.md](data-prep/eda.md) file.