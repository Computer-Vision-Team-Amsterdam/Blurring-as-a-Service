from pathlib import Path
import json
from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath
from azureml.core import Workspace
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient


SUBSCRIPTION_ID = 'b5d1b0e0-1ce4-40f9-87d5-cf3fde7a7b14'
RESOURCE_GROUP = 'cvo-aml-p-rg'
WORKSPACE_NAME = 'cvo-weu-aml-p-xnjyjutinwfyu'

CONTAINER_NAME = 'annotations-blob-container'
STORAGE_ACCOUNT = 'https://cvodataweupgwapeg4pyiw5e.blob.core.windows.net/'

def get_annotated_pano_ids(annotations_file):
    f = open(annotations_file)
    annotated_files = json.load(f)
    pano_ids = []
    for img in annotated_files['images']:
        pano_id = img['file_name']
        pano_ids.append(pano_id)
    return pano_ids


def remove_images_from_datastore(pano_ids):
    default_credential = DefaultAzureCredential()
    to_be_deleted = []
    files_found = 0
    # Create the BlobServiceClient object
    blob_service_client = ContainerClient(STORAGE_ACCOUNT, container_name=CONTAINER_NAME, credential=default_credential)
    for file in blob_service_client.list_blob_names():
        if file not in pano_ids and "blurring-project" in file and file[-4:] == ".jpg":
            to_be_deleted.append(file)
        if file in pano_ids:
            files_found +=1
    # blob_service_client.delete_blob(to_be_deleted[0])


def update_dataset():
    workspace = Workspace(SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
    dataset = Dataset.get_by_name(workspace, name='ann-blurring-dataset')
    dataset.update()


def remove_images(annotations_file):
    annotated_panos = get_annotated_pano_ids(annotations_file)
    remove_images_from_datastore(annotated_panos)
    update_dataset()


if __name__ == "__main__":
    annotations_file = Path("data/prep-laurens/annotations.json")
    remove_images(annotations_file)