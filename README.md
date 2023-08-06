# Blurring-as-a-Service

This project is about removing personal data, i.e. persons and license plates from raw panorama images.
We aim to do this in an inclusive manner, and we select our images based on different types of biases which can occur.
We create a document with a list of potential biases that we want to minimize and we select 
the panorama images used for training the model based on this document.

### Useful links
1. [Panorama API](https://api.data.amsterdam.nl/panorama/panoramas)
2. [Panorama viewer](https://data.amsterdam.nl/data/geozoek/?modus=kaart&term=Panoramabeelden&lagen=pano-pano2022bi%7Cpano-pano2021bi%7Cpano-pano2020bi%7Cpano-pano2019bi%7Cpano-pano2018bi%7Cpano-pano2017bi%7Cpano-pano2016bi%7Cpano-pano2021woz%7Cpano-pano2020woz%7Cpano-pano2019woz%7Cpano-pano2018woz%7Cpano-pano2017woz&legenda=true)
3. [Excel sheet with inclusivity biases (risico's en maatregelen.xlsx)](https://hoofdstad.sharepoint.com/sites/DigitaliseringenCTO/Shared%20Documents/Forms/AllItems.aspx?RootFolder=%2Fsites%2FDigitaliseringenCTO%2FShared%20Documents%2FInnovatie%20en%20RenD%2FComputer%20Vision%20Team%2FProjecten%2FInnovatiebudget%20%28hieronder%20valt%20Blur%20use%20case%29%2FInclusiviteit&FolderCTID=0x0120002EC45AFB501BC64FB525D14106AF3E05)
4. [Annotation project in Azure ML](https://ml.azure.com/labeling/project/93e9b2be-62de-6a8c-9c22-5b20cc5b90af/details?wsid=/subscriptions/b5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14/resourceGroups/cvo-aml-p-rg/providers/Microsoft.MachineLearningServices/workspaces/cvo-weu-aml-p-xnjyjutinwfyu&tid=72fca1b1-2c2e-4376-a445-294d80196804)
5. [Miro board with pipelines architecture](https://miro.com/app/board/uXjVPbDfQ9s=/?share_link_id=940866715023)
---

## Development

#### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/Blurring-as-a-Service.git
```

#### 2. Install Poetry
If you don't have it yet, follow the instructions [here](https://python-poetry.org/docs/#installation) to install the package manager Poetry.


#### 3. Init submodules
You need to initialize the content of the submodules so git clones the latest version.
```bash
git submodule update --init --recursive
```

#### 4. Setup the AzureML connection
To allow your code to connect to Azure ML and train the model is necessary to retrieve a connection config.
This can be done clicking on the change workspace button located on the top right in the [AzureML website](https://ml.azure.com), and then in "Download config file".
The downloaded "config.json" file must be added in the top folder of the project.

#### 5. Install dependencies
In the terminal, navigate to the project root (the folder containing `pyproject.toml`), then use Poetry to create a new virtual environment and install the dependencies.

```bash
poetry install
```
    
#### 6. Install pre-commit hooks
The pre-commit hooks help to ensure that all committed code is valid and consistently formatted.

```bash
poetry run pre-commit install
```

#### 7. Setup running configuration
Copy the config.example.yml file and rename it to config.yml.
Adapt the config file to your execution configuration, 
setting azure paths of where the data can be located and the flags to enable or disable pipeline steps.

---

### Update yolov5
In case the yolov5 package gets updated it's necessary to update the poetry packages adding the new ones of yolov5,
in case there are new ones.

This can be easily done using the following command:
```bash
cat yolov5/requirements.txt | grep -o '^[^# ]*' | xargs poetry add
```

---

### Pipelines
More information about the pipelines can be found [here](docs/performance_evaluation_pipeline.md).