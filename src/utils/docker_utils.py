"""
Small utilities used at container/app startup to fail fast with clear
messages instead of the Streamlit app silently crashing.
"""
import shutil
import urllib.request

from src.utils.config import DATASET_URL, MODEL_ARTIFACT_PATH, RAW_TRAIN_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _download_dataset() -> None:
    """Downloads application_train.csv from DATASET_URL (e.g. a GitHub
    Release asset) to RAW_TRAIN_FILE. Used for public deployments where the
    dataset can't be committed to git and can't be placed manually."""
    RAW_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = RAW_TRAIN_FILE.with_suffix(".csv.partial")
    logger.info("Downloading dataset from DATASET_URL...")
    try:
        with urllib.request.urlopen(DATASET_URL) as response, open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
        tmp_path.rename(RAW_TRAIN_FILE)
        size_mb = RAW_TRAIN_FILE.stat().st_size / (1024 * 1024)
        logger.info("Downloaded application_train.csv (%.1f MB) -> %s", size_mb, RAW_TRAIN_FILE)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def check_dataset_present() -> bool:
    if not RAW_TRAIN_FILE.exists():
        if DATASET_URL:
            try:
                _download_dataset()
            except Exception as e:
                logger.error("Dataset download from DATASET_URL failed: %s", e)
                return False
        else:
            logger.warning(
                "Dataset not found at %s and DATASET_URL is not set. Mount the "
                "Home Credit CSVs into ./data (see .env.example / README).",
                RAW_TRAIN_FILE,
            )
            return False
    return RAW_TRAIN_FILE.exists()


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