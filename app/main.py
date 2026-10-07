from io import BytesIO
from pathlib import Path
import base64

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
)
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

from app.schemas import (
    Detection,
    PredictionResponse,
    SampleImage,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fod_yolo_baseline_best.pt"
)


SAMPLE_DIR = (
    PROJECT_ROOT
    / "examples"
    / "sample_images"
)


PROCESSED_TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "yolo_dataset"
    / "images"
    / "test"
)


RAW_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "FODPascalVOCFormat-V.2.1"
    / "VOC2007"
    / "JPEGImages"
)


ALLOWED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


model = YOLO(
    str(MODEL_PATH)
)


app = FastAPI(
    title="AI Runway FOD Detection API",
    description=(
        "Computer vision prototype for "
        "Foreign Object Debris detection."
    ),
    version="1.0.0",
)


APP_DIR = Path(
    __file__
).resolve().parent


app.mount(
    "/static",
    StaticFiles(
        directory=APP_DIR / "static"
    ),
    name="static",
)


def find_image_file(
    filename: str
):

    safe_name = Path(
        filename
    ).name

    if safe_name != filename:
        return None

    candidate_directories = [
        SAMPLE_DIR,
        PROCESSED_TEST_DIR,
        RAW_IMAGE_DIR,
    ]

    for directory in candidate_directories:

        candidate = (
            directory
            / safe_name
        )

        if candidate.is_file():

            if (
                candidate.suffix.lower()
                in ALLOWED_SUFFIXES
            ):

                return candidate

    return None


def image_to_data_url(
    image_array
):

    image = Image.fromarray(
        image_array
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def predict_image(
    image: Image.Image,
    filename: str,
    confidence: float,
):

    image = image.convert(
        "RGB"
    )

    image_width, image_height = (
        image.size
    )

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        boxes = result.boxes

        for index in range(
            len(boxes)
        ):

            class_id = int(
                boxes.cls[index].item()
            )

            confidence_score = float(
                boxes.conf[index].item()
            )

            xmin, ymin, xmax, ymax = (
                boxes.xyxy[index].tolist()
            )

            class_name = model.names.get(
                class_id,
                str(class_id),
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence_score,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                )
            )

    detections.sort(
        key=lambda item: item.confidence,
        reverse=True,
    )

    if detections:

        status = (
            "FOD DETECTED - REVIEW REQUIRED"
        )

    else:

        status = (
            "NO FOD DETECTED "
            "AT CURRENT THRESHOLD"
        )

    annotated_bgr = result.plot()

    annotated_rgb = (
        annotated_bgr[:, :, ::-1]
    )

    return PredictionResponse(
        filename=filename,
        status=status,
        detection_count=len(detections),
        image_width=image_width,
        image_height=image_height,
        detections=detections,
        annotated_image=image_to_data_url(
            annotated_rgb
        ),
    )


@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():

    html_path = (
        APP_DIR
        / "templates"
        / "index.html"
    )

    return html_path.read_text(
        encoding="utf-8"
    )


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(
            MODEL_PATH
        ),
    }


@app.get(
    "/samples",
    response_model=list[SampleImage],
)
def samples():

    sample_files = []

    if SAMPLE_DIR.exists():

        for path in sorted(
            SAMPLE_DIR.iterdir()
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in ALLOWED_SUFFIXES
            ):
                continue

            sample_files.append(
                SampleImage(
                    filename=path.name,
                    url=f"/samples/{path.name}",
                )
            )

    return sample_files


@app.get(
    "/samples/{filename}"
)
def sample_file(
    filename: str
):

    image_path = find_image_file(
        filename
    )

    if image_path is None:

        raise HTTPException(
            status_code=404,
            detail="Sample image not found.",
        )

    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
    confidence: float = Query(
        0.25,
        ge=0.05,
        le=0.95,
    ),
):

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Upload JPG, PNG, or WEBP."
            ),
        )

    try:

        file_bytes = await file.read()

        image = Image.open(
            BytesIO(file_bytes)
        ).convert("RGB")

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read "
                "the uploaded image."
            ),
        ) from exc

    return predict_image(
        image=image,
        filename=(
            file.filename
            or "uploaded_image"
        ),
        confidence=confidence,
    )


@app.post(
    "/predict-sample/{filename}",
    response_model=PredictionResponse,
)
def predict_sample(
    filename: str,
    confidence: float = Query(
        0.25,
        ge=0.05,
        le=0.95,
    ),
):

    image_path = find_image_file(
        filename
    )

    if image_path is None:

        raise HTTPException(
            status_code=404,
            detail="Sample image not found.",
        )

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail="Could not read sample image.",
        ) from exc

    return predict_image(
        image=image,
        filename=filename,
        confidence=confidence,
    )