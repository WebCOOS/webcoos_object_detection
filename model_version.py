from enum import Enum


class ModelFramework(str, Enum):
    YOLO = "YOLO"

class YOLOModelName(str, Enum):
    best_yolo = "best_yolo"


class YOLOModelVersion(str, Enum):
    one = "1"
