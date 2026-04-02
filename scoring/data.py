"""Load raw subsidy register export."""

from __future__ import annotations

import pandas as pd

from .config import DATA_PATH

HEADER_ROW = 4  # 0-based: fifth row in Excel = headers per ISS export


def load_raw_excel(path: str | None = None) -> pd.DataFrame:
    p = path or str(DATA_PATH)
    df = pd.read_excel(p, header=HEADER_ROW, engine="openpyxl")
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.rename(
        columns={
            "№ п/п": "row_id",
            "Дата поступления": "application_datetime",
            "Область": "region",
            "Акимат": "akimat",
            "Номер заявки": "application_id",
            "Направление водства": "livestock_direction",
            "Наименование субсидирования": "subsidy_program_name",
            "Статус заявки": "application_status",
            "Норматив": "normative",
            "Причитающая сумма": "eligible_amount_kzt",
            "Район хозяйства": "farm_district",
        }
    )
    return df


def add_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds outcome_positive and exclude_from_training."""
    from .config import EXCLUDED_STATUSES, NEGATIVE_STATUSES, POSITIVE_STATUSES

    out = df.copy()
    s = out["application_status"].astype(str).str.strip()
    out["outcome_positive"] = s.map(
        lambda x: 1
        if x in POSITIVE_STATUSES
        else (0 if x in NEGATIVE_STATUSES else float("nan"))
    )
    out["exclude_from_training"] = s.isin(EXCLUDED_STATUSES)
    return out
