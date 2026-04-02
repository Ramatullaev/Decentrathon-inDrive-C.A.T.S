"""Train XGBoost classifier, evaluate, save artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    EXCLUDED_STATUSES,
    NEGATIVE_STATUSES,
    POSITIVE_STATUSES,
    RANDOM_STATE,
)
from .data import add_target_column, load_raw_excel
from .features import build_feature_matrix


def prepare_training_data(
    excel_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = load_raw_excel(excel_path)
    df = add_target_column(df)
    mask = ~df["exclude_from_training"] & df["outcome_positive"].notna()
    df_fit = df.loc[mask].reset_index(drop=True)
    y = df_fit["outcome_positive"].astype(int)
    X, feature_cols = build_feature_matrix(df_fit)
    return df_fit, X, y


def train_and_save(
    excel_path: str | None = None,
    artifacts_dir: Path | None = None,
) -> dict:
    out_dir = artifacts_dir or ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df_fit, X, y = prepare_training_data(excel_path)
    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=float(scale),
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "positive_rate_train": float(y_train.mean()),
    }

    joblib.dump(model, out_dir / "xgb_model.joblib")
    joblib.dump(feature_cols, out_dir / "feature_columns.joblib")

    meta = {
        "feature_columns": feature_cols,
        "metrics": metrics,
        "target_definition": {
            "positive_statuses": sorted(POSITIVE_STATUSES),
            "negative_statuses": sorted(NEGATIVE_STATUSES),
            "excluded_statuses": sorted(EXCLUDED_STATUSES),
        },
    }
    with open(out_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Feature importance for quick UI without SHAP
    imp = dict(zip(feature_cols, model.feature_importances_.tolist()))
    with open(out_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(imp.items(), key=lambda x: -x[1])), f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    p = path or str(DATA_PATH)
    print(f"Training on: {p}")
    metrics = train_and_save(excel_path=path)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
