from enum import Enum


class ModelFramework(str, Enum):
    ultralytics = "ultralytics"
    sahi = "sahi"


class YOLOModelName(str, Enum):
    yolo = "yolo"


class YOLOModelVersion(str, Enum):
    v8n = "v8n"


# SAHI models / versions will track what's defined for YOLO, as the SAHI
# method re-uses the other models.
class SAHIModelName(str, Enum):
    yolo = "yolo"


class SAHIModelVersion(str, Enum):
    v8n = "v8n"


class SAHISliceSize(int, Enum):
    slice_128   = 128
    slice_256   = 256
    slice_512   = 512
    slice_1024  = 1024


class YOLOModelObjectClassification(str, Enum):
    '''Limited list of the object types (possibly) relevant to WebCOOS.'''

    person = 'person'
    bicycle = 'bicycle'
    car = 'car'
    motorcycle = 'motorcycle'
    airplane = 'airplane'
    bus = 'bus'
    train = 'train'
    truck = 'truck'
    boat = 'boat'
    # traffic_light = 'traffic light'
    # fire_hydrant = 'fire hydrant'
    # stop_sign = 'stop sign'
    # parking_meter = 'parking meter'
    # bench = 'bench'
    bird = 'bird'
    cat = 'cat'
    dog = 'dog'
    horse = 'horse'
    sheep = 'sheep'
    cow = 'cow'
    # elephant = 'elephant'
    bear = 'bear'
    # zebra = 'zebra'
    # giraffe = 'giraffe'
    # backpack = 'backpack'
    umbrella = 'umbrella'
    # handbag = 'handbag'
    # tie = 'tie'
    # suitcase = 'suitcase'
    # frisbee = 'frisbee'
    # skis = 'skis'
    # snowboard = 'snowboard'
    sports_ball = 'sports ball'
    kite = 'kite'
    # baseball_bat = 'baseball bat'
    # baseball_glove = 'baseball glove'
    # skateboard = 'skateboard'
    surfboard = 'surfboard'
    # tennis_racket = 'tennis racket'
    # bottle = 'bottle'
    # wine_glass = 'wine glass'
    # cup = 'cup'
    # fork = 'fork'
    # knife = 'knife'
    # spoon = 'spoon'
    # bowl = 'bowl'
    # banana = 'banana'
    # apple = 'apple'
    # sandwich = 'sandwich'
    # orange = 'orange'
    # broccoli = 'broccoli'
    # carrot = 'carrot'
    # hot_dog = 'hot dog'
    # pizza = 'pizza'
    # donut = 'donut'
    # cake = 'cake'
    # chair = 'chair'
    # couch = 'couch'
    # potted_plant = 'potted plant'
    # bed = 'bed'
    # dining_table = 'dining table'
    # toilet = 'toilet'
    # tv = 'tv'
    # laptop = 'laptop'
    # mouse = 'mouse'
    # remote = 'remote'
    # keyboard = 'keyboard'
    # cell_phone = 'cell phone'
    # microwave = 'microwave'
    # oven = 'oven'
    # toaster = 'toaster'
    # sink = 'sink'
    # refrigerator = 'refrigerator'
    # book = 'book'
    # clock = 'clock'
    # vase = 'vase'
    # scissors = 'scissors'
    # teddy_bear = 'teddy bear'
    # hair_drier = 'hair drier'
    # toothbrush = 'toothbrush'
