"""Paths and settings used by the notebooks, predict.py, and the API."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
RG_MODEL_DIR = MODELS_DIR / "rg"
BK_MODEL_DIR = MODELS_DIR / "bk"
RG_HUB_ID = "newnus/justid-rechtsgebieden"
BK_HUB_ID = "newnus/justid-bijzondere-kenmerken"
MAX_LEN = 512
LABEL_THRESHOLD = 0.5

MIN_TEXT_LENGTH = 500
MIN_LABEL_FREQUENCY = 100
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
RANDOM_SEED = 42
