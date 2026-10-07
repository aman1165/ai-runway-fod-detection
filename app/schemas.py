from pydantic import BaseModel, Field


class Detection(BaseModel):

    class_id: int

    class_name: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    xmin: float
    ymin: float
    xmax: float
    ymax: float


class PredictionResponse(BaseModel):

    filename: str

    status: str

    detection_count: int

    image_width: int

    image_height: int

    detections: list[Detection]

    annotated_image: str


class SampleImage(BaseModel):

    filename: str

    url: str