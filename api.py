import os
# import requests
from typing import Any, List
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from yolo_processing import yolo_process_image, YOLO_MODELS
from sahi_processing import sahi_process_image, SAHI_MODELS
from metrics import make_metrics_app
from namify import namify_for_content
from score import ClassificationModelResult
from model_version import (
    YOLOModelName,
    YOLOModelVersion,
    YOLOModelObjectClassification,
    SAHIModelName,
    SAHIModelVersion,
    SAHISliceSize
)
import logging
from datetime import datetime, timezone

logger = logging.getLogger( __name__ )


app: FastAPI = FastAPI()
# Prometheus metrics
metrics_app = make_metrics_app()
app.mount("/metrics", metrics_app)


class UrlParams(BaseModel):
    url: str


ULTRALYTICS_ENDPOINT_PREFIX = "/ultralytics"
SAHI_ENDPOINT_PREFIX = "/sahi"
ALLOWED_IMAGE_EXTENSIONS = (
    "jpg",
    "png"
)

output_path = Path(os.environ.get(
    "OUTPUT_DIRECTORY",
    str(Path(__file__).with_name('outputs') / 'fastapi')
))


def get_yolo_model(model: YOLOModelName, version: YOLOModelVersion):
    return YOLO_MODELS[model.value][version.value]


def get_sahi_model(model: SAHIModelName, version: SAHIModelVersion):
    return SAHI_MODELS[model.value][version.value]


# Mounting the 'static' output files for the app
app.mount(
    "/outputs",
    StaticFiles(directory=output_path),
    name="outputs"
)


def annotation_image_and_classification_result(
    url: str,
    classification_result: ClassificationModelResult
):

    dt = datetime.utcnow().replace( tzinfo=timezone.utc )
    dt_str = dt.isoformat( "T", "seconds" ).replace( '+00:00', 'Z' )

    return {
        "time": dt_str,
        "annotated_image_url": url,
        "classification_result": classification_result
    }


@app.get("/", include_in_schema=False)
async def index():
    """Convenience redirect to OpenAPI spec UI for service."""
    return RedirectResponse("/docs")


CLS_NAMES_VALID_OPENAPI_EXTRA = {
    'requestBody': {
        'content': {
            'multipart/form-data': {
                'encoding': {
                    'cls_names_valid': {
                        'explode' : True
                    }
                }
            }
        }
    }
}


# YOLO object detection endpoints
@app.post(
    f"{ULTRALYTICS_ENDPOINT_PREFIX}/{{model}}/{{version}}/upload",
    tags=['ultralytics'],
    summary="Ultralytics/YOLOv8 model prediction on image upload",
    openapi_extra=CLS_NAMES_VALID_OPENAPI_EXTRA
)
def yolo_from_upload(
    request: Request,
    model: YOLOModelName,
    version: YOLOModelVersion,
    file: UploadFile,
    confidence_threshold: float = Form( gt=0.0, lt=1.0, default=None ),
    cls_names_valid: List[YOLOModelObjectClassification] = Form( default=None ),
    yolo: Any = Depends(get_yolo_model),
):
    """Perform model prediction based on selected YOLOv8 model / version."""
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    ( res_path, classification_result) = yolo_process_image(
        yolo,
        output_path,
        model,
        version,
        name,
        bytedata
    )

    if( res_path is None ):
        return annotation_image_and_classification_result(
            None,
            classification_result
        )

    rel_path = os.path.relpath( res_path, output_path )

    url_path_for_output = rel_path

    try:
        # Try for an absolute URL (prefixed with http(s)://hostname, etc.)
        url_path_for_output = str( request.url_for( 'outputs', path=rel_path ) )
    except Exception:
        # Fall back to the relative URL determined by the router
        url_path_for_output = app.url_path_for(
            'outputs', path=rel_path
        )
    finally:
        pass

    return annotation_image_and_classification_result(
        url_path_for_output,
        classification_result
    )


# SAHI-related object detection endpoints
@app.post(
    f"{SAHI_ENDPOINT_PREFIX}/{{model}}/{{version}}/upload",
    tags=['sahi'],
    summary="SAHI-wrapped Ultralytics/YOLOv8 model prediction on image upload",
    openapi_extra=CLS_NAMES_VALID_OPENAPI_EXTRA
)
def sahi_from_upload(
    request: Request,
    model: SAHIModelName,
    version: SAHIModelVersion,
    file: UploadFile,
    confidence_threshold: float = Form( gt=0.0, lt=1.0, default=None ),
    cls_names_valid: List[YOLOModelObjectClassification] = Form( default=None ),
    slice_size: SAHISliceSize = Form( default=SAHISliceSize.slice_512 ),
    yolo: Any = Depends(get_sahi_model),
):
    """Perform model prediction based on selected SAHI + YOLOv8 model / """
    """version."""
    bytedata = file.file.read()

    ( name, ext ) = namify_for_content( bytedata )

    assert ext in ALLOWED_IMAGE_EXTENSIONS, \
        f"{ext} not in allowed image file types: {repr(ALLOWED_IMAGE_EXTENSIONS)}"

    ( res_path, classification_result) = sahi_process_image(
        yolo,
        output_path,
        model,
        version,
        name,
        bytedata,
        confidence_threshold,
        slice_size,
        cls_names_valid
    )

    if( res_path is None ):
        return annotation_image_and_classification_result(
            None,
            classification_result
        )

    rel_path = os.path.relpath( res_path, output_path )

    url_path_for_output = rel_path

    try:
        # Try for an absolute URL (prefixed with http(s)://hostname, etc.)
        url_path_for_output = str( request.url_for( 'outputs', path=rel_path ) )
    except Exception:
        # Fall back to the relative URL determined by the router
        url_path_for_output = app.url_path_for(
            'outputs', path=rel_path
        )
    finally:
        pass

    return annotation_image_and_classification_result(
        url_path_for_output,
        classification_result
    )


@app.post("/health")
def health():
    return { "health": "ok" }
