import json

from blurring_as_a_service.utils.azure_coco_to_coco_converter import (
    AzureCocoToCocoConverter,
)


def compare_json_files(file1, file2):
    with open(file1, "r") as f1:
        data1 = json.load(f1)
    with open(file2, "r") as f2:
        data2 = json.load(f2)

    if len(data1["images"]) != len(data2["images"]) or len(data1["annotations"]) != len(
        data2["annotations"]
    ):
        return False

    for i in range(len(data1["images"])):
        if data1["images"][i]["id"] != data2["images"][i]["id"]:
            raise ValueError(
                f"Different ids: {data1['images'][i]['id']} is not {data2['images'][i]['id']}"
            )
        if data1["images"][i]["width"] != data2["images"][i]["width"]:
            raise ValueError(
                f"Different widths: {data1['images'][i]['width']} is not {data2['images'][i]['width']}"
            )
        if data1["images"][i]["height"] != data2["images"][i]["height"]:
            raise ValueError(
                f"Different heights: {data1['images'][i]['height']} is not {data2['images'][i]['height']}"
            )

    for i in range(len(data1["annotations"])):
        if data1["annotations"][i]["image_id"] != data2["annotations"][i]["image_id"]:
            raise ValueError(
                f"Different image_ids: {data1['annotations'][i]['image_id']} is not "
                f"{data2['annotations'][i]['image_id']}"
            )
        if (
            "iscrowd" not in data1["annotations"][i]
            or "iscrowd" not in data2["annotations"][i]
        ):
            raise ValueError("Missing argument iscrowd")

        if (
            "area" not in data1["annotations"][i]
            or "area" not in data2["annotations"][i]
        ):
            raise ValueError("Missing argument area")

    categories1 = set([d["name"] for d in data1["categories"]])
    categories2 = set([d["name"] for d in data2["categories"]])
    if categories1 != categories2:
        raise ValueError(f"Mismatched categories: {categories1} is not {categories2}")

    return True


def test_azurecoco_to_coco_untagged_data():
    source = "../../local_test_data/sample/azure-coco-format-for-sample-validation.json"
    output_file = "sample_validation__AzureCocoToCoco_converted.json"
    expected_file = "../../local_test_data/sample/coco_format_sample_val.json"

    converter = AzureCocoToCocoConverter(source, output_file)
    converter.convert()

    assert compare_json_files(output_file, expected_file)
