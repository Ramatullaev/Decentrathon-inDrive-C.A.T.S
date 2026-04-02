"""Feature engineering + simulated merit/compliance features (see README)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


def _stable_seed(application_id: float | int | str, salt: str = "") -> int:
    raw = f"{application_id}|{salt}"
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["application_datetime"], errors="coerce", dayfirst=True)
    out["application_month"] = dt.dt.month.fillna(-1).astype(int)
    out["application_dow"] = dt.dt.dayofweek.fillna(-1).astype(int)
    out["days_since_year_start"] = dt.dt.dayofyear.fillna(-1).astype(int)

    out["normative_numeric"] = pd.to_numeric(out["normative"], errors="coerce")
    out["eligible_amount_numeric"] = pd.to_numeric(out["eligible_amount_kzt"], errors="coerce")
    out["amount_per_normative"] = np.where(
        (out["normative_numeric"].notna()) & (out["normative_numeric"] != 0),
        out["eligible_amount_numeric"] / out["normative_numeric"],
        np.nan,
    )
    out["log_eligible_amount"] = np.log1p(out["eligible_amount_numeric"].clip(lower=0))

    for col in ("region", "farm_district", "livestock_direction", "subsidy_program_name"):
        out[f"{col}_code"], _ = pd.factorize(out[col].astype(str), sort=True)

    day_key = dt.dt.strftime("%Y-%m-%d")
    dup = (
        out.groupby(["farm_district", day_key], dropna=False)["application_id"]
        .transform("count")
        .fillna(0)
    )
    out["duplicate_application_flag"] = (dup > 1).astype(int)

    return out


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-region aggregates (descriptive; see README)."""
    out = df.copy()
    g = out.groupby("region", dropna=False)["eligible_amount_numeric"]
    out["region_mean_amount"] = g.transform("mean")
    out["region_app_count"] = g.transform("count")
    return out


def add_simulated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulated features not present in ISS export: merit, vet, documents, tax.
    Deterministic per application_id; documented as simulated in README.
    """
    out = df.copy()
    n = len(out)

    years = np.zeros(n, dtype=float)
    farm_ha = np.zeros(n, dtype=float)
    head_eq = np.zeros(n, dtype=float)
    prior_n = np.zeros(n, dtype=float)
    prior_sum = np.zeros(n, dtype=float)
    util = np.zeros(n, dtype=float)
    vet_match = np.zeros(n, dtype=int)
    seller_ok = np.zeros(n, dtype=int)
    buyer_ok = np.zeros(n, dtype=int)
    doc_score = np.zeros(n, dtype=float)
    fraud_score = np.zeros(n, dtype=float)
    tax_ok = np.zeros(n, dtype=int)
    land_per_head = np.zeros(n, dtype=float)

    ids = out["application_id"].fillna(0).astype(float)
    regions = out["region_code"].fillna(0).astype(int)

    for i in range(n):
        seed = _stable_seed(ids.iloc[i], "sim")
        r = np.random.default_rng(seed)
        base = (regions.iloc[i] % 17) / 17.0
        years[i] = 3 + 12 * r.random() + 2 * base
        farm_ha[i] = 50 + 800 * r.random() + 100 * base
        head_eq[i] = 20 + 500 * r.random() + 50 * base
        prior_n[i] = r.integers(0, 6)
        prior_sum[i] = prior_n[i] * (100_000 + 900_000 * r.random())
        util[i] = r.uniform(0.4, 1.0)
        vet_match[i] = int(r.random() < 0.92)
        seller_ok[i] = int(r.random() < 0.88)
        buyer_ok[i] = int(r.random() < 0.95)
        doc_score[i] = r.uniform(0.3, 1.0)
        fraud_score[i] = r.uniform(0, 0.35)
        tax_ok[i] = int(r.random() < 0.91)
        land_per_head[i] = max(0.01, farm_ha[i] / max(1.0, head_eq[i]))

    out["years_in_agribusiness"] = years
    out["farm_size_ha"] = farm_ha
    out["livestock_head_equivalent"] = head_eq
    out["prior_subsidy_count_3y"] = prior_n
    out["prior_subsidy_total_kzt_3y"] = prior_sum
    out["subsidy_utilization_rate"] = util
    out["vet_registry_match"] = vet_match
    out["seller_verified"] = seller_ok
    out["buyer_verified"] = buyer_ok
    out["contract_completeness_score"] = doc_score
    out["doc_fraud_risk_score"] = fraud_score
    out["tax_compliance_flag"] = tax_ok
    out["land_area_per_head"] = land_per_head

    out["merit_index_heuristic"] = (
        0.25 * out["subsidy_utilization_rate"]
        + 0.15 * out["contract_completeness_score"]
        + 0.15 * (out["vet_registry_match"] + out["seller_verified"] + out["buyer_verified"]) / 3.0
        + 0.15 * out["tax_compliance_flag"]
        + 0.10 * np.clip(out["land_area_per_head"] / 5.0, 0, 1)
        + 0.10 * np.log1p(out["years_in_agribusiness"]) / 3.0
        + 0.10 * (1.0 - out["doc_fraud_risk_score"])
    )

    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Returns X with only numeric model columns and column list."""
    df = add_engineered_features(df)
    df = add_aggregate_features(df)
    df = add_simulated_features(df)

    feature_cols = [
        "application_month",
        "application_dow",
        "days_since_year_start",
        "normative_numeric",
        "eligible_amount_numeric",
        "amount_per_normative",
        "log_eligible_amount",
        "region_code",
        "farm_district_code",
        "livestock_direction_code",
        "subsidy_program_name_code",
        "duplicate_application_flag",
        "region_mean_amount",
        "region_app_count",
        "years_in_agribusiness",
        "farm_size_ha",
        "livestock_head_equivalent",
        "prior_subsidy_count_3y",
        "prior_subsidy_total_kzt_3y",
        "subsidy_utilization_rate",
        "vet_registry_match",
        "seller_verified",
        "buyer_verified",
        "contract_completeness_score",
        "doc_fraud_risk_score",
        "tax_compliance_flag",
        "land_area_per_head",
        "merit_index_heuristic",
    ]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X, feature_cols
