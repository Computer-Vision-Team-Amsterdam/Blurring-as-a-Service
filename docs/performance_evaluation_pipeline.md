## Performance evaluation pipeline

`experiment_name`

when we run yolo we set a `--name` flag with the name of the folder where
the output is stored, e.g. `exp`. 
If we re-run yolo without adjusting the name of the experiment, then 
`exp_2` is automatically created.
Subsequent tasks, i.e. calculation of metrics will compute metrics based on 
`exp`. With `experiment_name` we want to point explicitly to which output folder is used throughout the tasks.


Notes on annotation formats:

Azure COCO format, COCO format for evaluation with yolov5


Differences between COCO format and Azure COCO format

- Miscalculation of area

AzureML COCO file contains area of 0 for all images. Instead, it should contain the area of the bbox
  
- Missing keys

AzureML COCO file does not contain the "iscrowd" and "segmentation" keys in the annotations,
which causes errors at evaluation time.


| COCO annotations for COCO evaluation with yolov5  | Azure COCO annotation                             |
|---------------------------------------------------|---------------------------------------------------|
| COCO format                                       | Azure COCO format with tagged classes             |  
| categories, indexed from 0                        | categories, indexed from 1                        |  
| categories are compatible with COCO evaluator     | categories are not compatible with COCO evaluator |
| absolute coordinates                              | normalized coordinates                            |  