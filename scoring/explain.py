"""SHAP-based explanations (top contributing features per application)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


def top_feature_reasons(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    row_index: int,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Returns top_k features by absolute SHAP value for a single row.
    Each item: {name, shap_value, feature_value, direction}
    """
    if shap is None:
        raise ImportError("shap is required for explanations")

    explainer = shap.TreeExplainer(model)
    row = X.iloc[[row_index]]
    sv = explainer.shap_values(row)[0]
    order = np.argsort(np.abs(sv))[::-1][:top_k]
    out = []
    for j in order:
        val = float(row.iloc[0, j])
        s = float(sv[j])
        out.append(
            {
                "name": feature_names[j],
                "shap_value": s,
                "feature_value": val,
                "direction": "increases_score" if s > 0 else "decreases_score",
            }
        )
    return out


def batch_top_reasons(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 3,
    max_samples: int = 500,
) -> list[list[dict[str, Any]]]:
    """SHAP explanations for up to max_samples rows (for export / audit)."""
    if shap is None:
        raise ImportError("shap is required for explanations")

    Xs = X.iloc[:max_samples]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    results = []
    for i in range(len(Xs)):
        order = np.argsort(np.abs(sv[i]))[::-1][:top_k]
        row = []
        for j in order:
            row.append(
                {
                    "name": feature_names[j],
                    "shap_value": float(sv[i][j]),
                    "feature_value": float(Xs.iloc[i, j]),
                    "direction": "increases_score" if sv[i][j] > 0 else "decreases_score",
                }
            )
        results.append(row)
    return results
