import os
from typing import Dict, List, Set, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from score import ClassificationModelResult, BoundingBoxPoint
from model_version import (
    ModelFramework,
    SAHIModelName,
    SAHIModelVersion,
    SAHISliceSize,
    YOLOModelObjectClassification
)
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from metrics import increment_detection_counter, increment_object_counter
from yolo_processing import YOLO_MODELS
import logging

logger = logging.getLogger( __name__ )
width = 896
height = 896
DEFAULT_SAHI_THRESHOLD = 0.5
DEFAULT_SAHI_OVERLAP_RATIO = 0.2
font = cv2.FONT_HERSHEY_SIMPLEX

MODEL_FOLDER = Path(os.environ.get(
    "MODEL_DIRECTORY",
    str(Path(__file__).parent)
))

# The SAHI library is a sub-sampling library that can 'wrap' a number of other
# ML processes. In this case, we are using it to wrap and sub-sample our YOLO
# models in the interest of getting better results for some use cases (lots of
# birds in a single image, for example).

SAHI_MODELS = {
    # public-facing model name
    "yolo": {
        # public-facing model version
        "v8n": YOLO_MODELS['yolo']['v8n']
    }
}


def sahi_process_image(
    yolo_model: YOLO,
    output_path: Path,
    model: Union[SAHIModelName, str],
    version: Union[SAHIModelVersion, str],
    name: str,
    bytedata: bytes,
    confidence_threshold: float = None,
    slice_size: Union[SAHISliceSize, int] = SAHISliceSize.slice_512,
    cls_names_valid: List[YOLOModelObjectClassification] = None,
):

    assert yolo_model, \
        f"Must have yolo_model passed to {sahi_process_image.__name__}"

    assert output_path and isinstance( output_path, Path ), \
        f"output_path parameter for {sahi_process_image.__name__} is not Path"

    assert output_path.exists() and output_path.is_dir(), \
        (
            f"output_path parameter for {sahi_process_image.__name__} must exist "
            "and be a directory"
        )

    assert isinstance( model, ( SAHIModelName, str ) )
    assert isinstance( version, ( SAHIModelVersion, str ) )

    if( isinstance( model, SAHIModelName ) ):
        model = model.value

    if( isinstance( version, SAHIModelVersion ) ):
        version = version.value

    if confidence_threshold is None:
        confidence_threshold = DEFAULT_SAHI_THRESHOLD

    assert confidence_threshold >= 0.0 and confidence_threshold <= 1.0

    if (
        cls_names_valid is None
        or len( [ x for x in cls_names_valid if x ]) <= 0
    ):
        # Default to empty list
        cls_names_valid = list( YOLOModelObjectClassification )

    assert all(
        [
            isinstance( x, YOLOModelObjectClassification )
            for x in cls_names_valid
        ]
    )

    if slice_size is None:
        slice_size = SAHISliceSize.slice_512

    # Use the actual int value associated with the enum
    slice_size = slice_size.value

    ret: ClassificationModelResult = ClassificationModelResult(
        ModelFramework.sahi.name,
        model,
        version
    )

    output_file = output_path / model / str(version) / name

    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_boxes = frame

    sahi_underlying_model_type_mapping: Dict[Tuple[str, str], str] = {
        ( SAHIModelName.yolo.value, SAHIModelVersion.v8n.value): 'yolov8',
    }

    # Throw a fit if we haven't seen this combination of model/version to hint
    # sahi at what type of model we're dealing with
    if ( model, version ) not in sahi_underlying_model_type_mapping:
        msg = (
            "Programmer error, unable to map model/version to SAHI underlying"
            f"model type: {model}/{version}"
        )
        logger.error( msg )
        raise Exception( msg )

    sahi_underlying_model_type = sahi_underlying_model_type_mapping[
        ( model, version, )
    ]

    sahi_wrapped_model = AutoDetectionModel.from_pretrained(
        model_type=sahi_underlying_model_type,
        model=yolo_model
    )

    prediction_result = get_sliced_prediction(
        frame,
        sahi_wrapped_model,
        slice_height = slice_size,
        slice_width = slice_size,
        overlap_height_ratio = DEFAULT_SAHI_OVERLAP_RATIO,
        overlap_width_ratio = DEFAULT_SAHI_OVERLAP_RATIO
    )

    # If any score is above threshold, flag it as detected
    detected = False

    yolo_object_class: YOLOModelObjectClassification = None

    object_classes_detected: Set[YOLOModelObjectClassification] = set()

    for prediction in prediction_result.object_prediction_list:
        bbox = prediction.bbox

        cls_name = prediction.category.name

        score = prediction.score.value
        if score < confidence_threshold:
            continue

        if cls_name is None:
            continue

        try:
            yolo_object_class = YOLOModelObjectClassification( cls_name )
        except ValueError:
            logger.warning(
                f"Detected class of '{cls_name}', but not among accepted "
                "classes, ignoring."
            )
            continue

        if yolo_object_class not in cls_names_valid:
            # Skip object, even if it meets with our threshold, as we aren't
            # interested for this particular detection
            continue

        detected = True

        ( x1, y1, x2, y2 ) = (
            int(bbox.minx),
            int(bbox.miny),
            int(bbox.maxx),
            int(bbox.maxy),
        )
        h, w, _ = frame.shape

        y_min = int(max(1, y1))
        x_min = int(max(1, x1))
        y_max = int(min(h, y2))
        x_max = int(min(w, x2))

        if yolo_object_class is not None:

            label = cls_name + ": " + ": {:.2f}%".format(score * 100)
            img_boxes = cv2.rectangle(img_boxes, (x_min, y_max), (x_max, y_min), (0, 0, 255), 2)
            cv2.putText(img_boxes, label, (x_min, y_max - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            ret.add(
                classification_name=yolo_object_class.value,
                classification_score=score,
                bbox=(
                    BoundingBoxPoint( x_min, y_min ),
                    BoundingBoxPoint( x_max, y_max ),
                )
            )

            # Update object metrics
            increment_object_counter(
                ModelFramework.sahi.name,
                model,
                version,
                yolo_object_class.value
            )

            # Track which of the object classes were detected for later
            # metrics counters
            object_classes_detected.add(
                yolo_object_class
            )

    # outp = cv2.resize(img_boxes, (1280, 720))

    if detected is True:

        if( object_classes_detected ):

            for ocd in object_classes_detected:

                # Update Prometheus metrics for each of the classes that were
                # detected
                increment_detection_counter(
                    ModelFramework.sahi.name,
                    model,
                    version,
                    ocd.value
                )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), img_boxes )
        return ( str(output_file), ret )

    return ( None, ret  )
