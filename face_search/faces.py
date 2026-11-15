"""Face detection + embeddings with DeepFace (shared by the pipeline and the search script)."""
from dataclasses import dataclass

import numpy as np
from deepface import DeepFace

# Choices benchmarked on our own images: see NOTES.md
MODEL_NAME = "Facenet512"
DETECTOR = "retinaface"
EMBEDDING_DIM = 512
# 0.9 let through a puma, hands holding a camera, a carved ornament (see README);
# 0.98 removes them for only 2% of the faces
MIN_CONFIDENCE = 0.98
# Smaller faces carry too little detail for a reliable embedding -> false matches
MIN_FACE_PX = 40


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    embedding: np.ndarray


def detect_and_embed(image_path: str) -> list[Face]:
    results = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        align=True,  # rotate the face upright using the eyes before embedding
        # Don't raise when no face is found (most images have none)
        enforce_detection=False,
    )
    faces = []
    for result in results:
        area = result["facial_area"]
        # With enforce_detection=False, "no face" comes back as the whole image with confidence 0
        if result["face_confidence"] < MIN_CONFIDENCE:
            continue
        if min(area["w"], area["h"]) < MIN_FACE_PX:
            continue
        faces.append(Face(
            x=area["x"], y=area["y"], w=area["w"], h=area["h"],
            confidence=float(result["face_confidence"]),
            embedding=np.asarray(result["embedding"], dtype=np.float32),
        ))
    return faces
