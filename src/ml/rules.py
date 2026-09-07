"""
Derives business-readable decision rules from the ML model.

Approach: fit a shallow (max_depth=3-4) decision tree as a *surrogate* on
the same transformed features and labels as the production LightGBM model.
A single 400-tree ensemble has no readable "rule" of its own, but a shallow
tree trained to mimic the data's decision boundary produces simple
if/then/else paths — e.g. "IF credit-to-income ratio > 5.2 AND age < 28
THEN High Risk (covers 4% of applicants, 31% default rate)" — that a credit
policy team can review, sanity-check, and turn into an actual policy,
independent of the black-box model's exact score.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _clean_feature_name(name: str) -> str:
    """Strips the sklearn ColumnTransformer 'num__' / 'cat__' prefixes for readability."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _tree_to_rules(tree_model: DecisionTreeClassifier, feature_names: list, y: pd.Series) -> list:
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = _clean_feature_name(feature_name[node])
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.2f}"])
        else:
            values = tree_.value[node][0]
            # tree_.value may hold raw counts or class proportions depending on
            # sklearn version/config — normalize defensively so this is correct either way.
            n_samples = int(tree_.n_node_samples[node])
            default_rate = float(values[1] / values.sum()) if values.sum() else 0.0
            if n_samples > 0:
                rules.append({
                    "conditions": conditions,
                    "n_samples": n_samples,
                    "coverage_pct": round(100 * n_samples / len(y), 2),
                    "default_rate_pct": round(100 * default_rate, 2),
                    "risk_label": (
                        "High" if default_rate >= 0.30 else "Medium" if default_rate >= 0.10 else "Low"
                    ),
                })

    recurse(0, [])
    rules.sort(key=lambda r: r["default_rate_pct"], reverse=True)
    return rules


def derive_rules(X: np.ndarray, y: pd.Series, feature_names: list, max_depth: int = 4) -> list:
    surrogate = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=max(50, int(0.01 * len(y))), random_state=42
    )
    surrogate.fit(X, y)
    rules = _tree_to_rules(surrogate, feature_names, y)
    logger.info("Derived %d business rules (surrogate tree depth=%d)", len(rules), max_depth)
    return rules


def rules_to_readable(rules: list) -> list:
    """Converts rule dicts into plain English sentences for the UI."""
    sentences = []
    for r in rules:
        cond_text = " AND ".join(r["conditions"]) if r["conditions"] else "All applicants"
        sentences.append(
            f"IF {cond_text} \u2192 {r['risk_label']} risk "
            f"(covers {r['coverage_pct']}% of applicants, {r['default_rate_pct']}% observed default rate)"
        )
    return sentences
