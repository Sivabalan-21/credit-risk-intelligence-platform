"""
Evaluation metrics for the credit default model.

Accuracy is intentionally NOT used as a headline metric: with an ~8% default
rate, a model that predicts "no default" for everyone scores ~92% accuracy
while being useless. ROC-AUC and PR-AUC (more informative under imbalance)
are the primary metrics, alongside recall at a business-relevant threshold
since missing a defaulter is costlier than a false alarm.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    tn, fp, fn, tp = np.array(cm).ravel()
    recall_defaulters = tp / (tp + fn) if (tp + fn) else 0.0
    precision_defaulters = tp / (tp + fp) if (tp + fp) else 0.0

    metrics = {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall_defaulters": round(float(recall_defaulters), 4),
        "precision_defaulters": round(float(precision_defaulters), 4),
        "classification_report": report,
    }
    logger.info("ROC-AUC=%.4f | PR-AUC=%.4f | Recall(default)=%.4f | Precision(default)=%.4f",
                roc_auc, pr_auc, recall_defaulters, precision_defaulters)
    return metrics


def get_roc_curve_points(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def get_pr_curve_points(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds
