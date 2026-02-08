from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
import os
import shutil

from metadata_layer import check_metadata
from inference.layer_b_infer import infer_image

app = FastAPI(
    title="AI Image Detection API",
    version="1.0"
)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG and PNG images allowed")

    image_bytes = await file.read()

    # ---------- Layer A: Metadata / C2PA ----------
    layer_a_result = check_metadata(image_bytes)

    if layer_a_result["is_ai"]:
        return {
            "is_ai_generated": True,
            "confidence": 1.0,
            "decision_layer": "layer_a_metadata",
            "reason": layer_a_result["reason"],
            "details": {
                "matched_keywords": layer_a_result.get("matched_keywords", [])
            }
        }

    # ---------- Layer B: CNN forensic detection ----------
    suffix = file.filename.split(".")[-1]
    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.{suffix}")

    try:
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        layer_b_result = infer_image(temp_path)

        return {
            "is_ai_generated": layer_b_result["is_ai_generated"],
            "confidence": round(layer_b_result["confidence"], 4),
            "decision_layer": "layer_b_cnn",
            "reason": (
                "No explicit AI provenance found; decision based on "
                "CNN-based forensic analysis"
            )
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
