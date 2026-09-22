"""FastAPI service around the two JustID BERTje classifiers.

Endpoints match the 2024 delivery: paste text, or upload a PDF.
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pdfplumber
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from predict import JustIdModels

_models: JustIdModels | None = None


class TextInput(BaseModel):
    text: str = Field(..., description="Dutch ruling text")


class PredictionOutput(BaseModel):
    rechtsgebieden: dict[str, float]
    bijzondere_kenmerken: dict[str, float]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _models
    _models = JustIdModels()
    yield
    _models = None


app = FastAPI(
    title="JustID",
    description="Multi-label rechtsgebieden and bijzondere kenmerken for Dutch rulings.",
    version="2026.0",
    lifespan=lifespan,
)


def _predict(text: str) -> PredictionOutput:
    if _models is None:
        raise HTTPException(status_code=503, detail="Models are not loaded")
    try:
        result = _models.predict(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionOutput(**result)


def _pdf_text(upload: UploadFile) -> str:
    name = (upload.filename or "").lower()
    if not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        pages: list[str] = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                chunk = page.extract_text()
                if chunk:
                    pages.append(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read the PDF") from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    text = "\n".join(pages).strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
    return text


@app.get("/")
def root() -> dict:
    return {
        "service": "JustID",
        "endpoints": {
            "predict_text": "POST /predict-text",
            "predict_document": "POST /predict-document",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health() -> dict:
    ready = _models is not None
    return {
        "status": "healthy" if ready else "unhealthy",
        "model_loaded": ready,
        "device": None if _models is None else str(_models.device),
    }


@app.post("/predict-text", response_model=PredictionOutput)
def predict_text(payload: TextInput) -> PredictionOutput:
    return _predict(payload.text)


@app.post("/predict-document", response_model=PredictionOutput)
def predict_document(file: UploadFile = File(...)) -> PredictionOutput:
    return _predict(_pdf_text(file))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
