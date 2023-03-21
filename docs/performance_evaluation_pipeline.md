## Performance evaluation pipeline



Notes

Differences between COCO format and Azure COCO format

- Miscalculation of area

AzureML COCO file contains area of 0 for all images. Instead, it should contain the area of the bbox
  
- Missing keys

AzureML COCO file does not contain the "iscrowd" key in the annotations, which causes errors at evaluation time.


How to obtain the custom_coco_file.json:

input: yolo annotations
run `python main.py --yolo-subdir --path /Users/epureanudiana/Documents/Projects/Yolo-to-COCO-format-converter/input --output test-14-mar.json
` in the YOLO-to-COCO-format-converter

run `python coco-to-customCoco.py` in the YOLO-to-COCO-format-converter


| custom_coco_categories_01.json                    | validation-tagged.json                            |
|---------------------------------------------------|---------------------------------------------------|
| COCO format                                       | Azure COCO format with tagged classes             |  
| categories, indexed from 0                        | categories, indexed from 1                        |  
| categories are not compatible with COCO evaluator | categories are not compatible with COCO evaluator |
| absolute coordinates                              | normalized coordinates                            |  