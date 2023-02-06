from enum import IntFlag, auto, unique
from functools import reduce


@unique
class PipelineFlag(IntFlag):
    NONE = 0
    CREATE_ENVIRONMENT = auto()
    CONVERT_COCO_TO_YOLO = auto()
    CREATE_METADATA = auto()
    GET_DATA = auto()

    @classmethod
    def all(cls):
        return reduce(lambda m1, m2: m1 | m2, [m for m in cls.__members__.values()])
