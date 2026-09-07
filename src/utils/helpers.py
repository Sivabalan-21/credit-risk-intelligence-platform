"""Small reusable helpers shared across modules."""
import json
from pathlib import Path
from typing import Any

from src.utils.config import RISK_LOW_MAX, RISK_MEDIUM_MAX


def risk_band(probability: float, low_max: float = RISK_LOW_MAX, medium_max: float = RISK_MEDIUM_MAX) -> str:
    """Map a predicted default probability to a business-readable risk band.
    Pass low_max/medium_max explicitly (loaded from models/risk_thresholds.json
    after training) to use data-driven, well-populated bands instead of the
    fixed fallback defaults — see config.py for why."""
    if probability < low_max:
        return "Low"
    if probability < medium_max:
        return "Medium"
    return "High"


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def format_currency(value: float) -> str:
    try:
        return f"${value:,.0f}"
    except (TypeError, ValueError):
        return str(value)