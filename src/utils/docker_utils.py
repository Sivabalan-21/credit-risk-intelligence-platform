"""
Small utilities used at container startup to fail fast with clear messages
instead of the Streamlit app silently crashing inside Docker.
"""
from pathlib import Path

from src.utils.config import RAW_TRAIN_FILE, MODEL_ARTIFACT_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_dataset_present() -> bool:
    if not RAW_TRAIN_FILE.exists():
        logger.warning(
            "Dataset not found at %s. Mount the Home Credit CSVs into ./data "
            "(see .env.example / README).",
            RAW_TRAIN_FILE,
        )
        return False
    return True


def check_model_trained() -> bool:
    if not MODEL_ARTIFACT_PATH.exists():
        logger.warning(
            "No trained model found at %s. Run `python -m src.ml.train` first, "
            "or use the 'Train model' button in the UI.",
            MODEL_ARTIFACT_PATH,
        )
        return False
    return True


def startup_checklist() -> dict:
    return {
        "dataset_present": check_dataset_present(),
        "model_trained": check_model_trained(),
    }
