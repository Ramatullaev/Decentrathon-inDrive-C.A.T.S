"""
Streamlit prototype: shortlist + merit score + SHAP top-3 reasons.
Run from project root: PYTHONPATH=. streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring.data import add_target_column, load_raw_excel
from scoring.explain import top_feature_reasons
from scoring.config import ARTIFACTS_DIR, DATA_PATH

ART_MODEL = ARTIFACTS_DIR / "xgb_model.joblib"
ART_COLS = ARTIFACTS_DIR / "feature_columns.joblib"


@st.cache_data(show_spinner=True)
def load_table(path_str: str) -> pd.DataFrame:
    return load_raw_excel(path_str if path_str else None)


@st.cache_resource
def load_model_bundle():
    if not ART_MODEL.exists() or not ART_COLS.exists():
        return None, None
    model = joblib.load(ART_MODEL)
    cols = joblib.load(ART_COLS)
    return model, cols


def main() -> None:
    st.set_page_config(page_title="Скоринг субсидий (демо)", layout="wide")
    st.title("Скоринг заявок на субсидии — shortlist для комиссии")
    st.caption(
        "Модель оценивает вероятность «положительного исхода» по историческим статусам; "
        "финальное решение остаётся за комиссией. Объяснения — SHAP по топ-3 признакам."
    )

    model, feature_cols = load_model_bundle()
    if model is None:
        st.error(
            "Артефакты модели не найдены. Выполните обучение: "
            "`PYTHONPATH=. .venv/bin/python -m scoring.train`"
        )
        return

    meta_path = DATA_PATH
    uploaded = st.sidebar.file_uploader("Загрузить Excel (опционально)", type=["xlsx"])
    if uploaded is not None:
        tmp = ROOT / "_uploaded_temp.xlsx"
        tmp.write_bytes(uploaded.getvalue())
        meta_path = str(tmp)

    df = load_table(str(meta_path))
    df = add_target_column(df)

    from scoring.features import build_feature_matrix

    X, cols = build_feature_matrix(df)
    if list(cols) != list(feature_cols):
        st.error("Список признаков не совпадает с обученной моделью. Запустите `python -m scoring.train` на этом файле.")
        return

    proba = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["merit_score"] = proba
    df["rank"] = df["merit_score"].rank(ascending=False, method="min").astype(int)

    regions = sorted(df["region"].dropna().astype(str).unique())
    programs = sorted(df["subsidy_program_name"].dropna().astype(str).unique())

    r_filter = st.sidebar.multiselect("Область", regions, default=[])
    p_filter = st.sidebar.multiselect("Программа субсидирования", programs, default=[])
    top_n = st.sidebar.slider("Размер shortlist", 10, 200, 50)
    min_score = st.sidebar.slider("Минимальный score", 0.0, 1.0, 0.35, 0.01)

    view = df
    if r_filter:
        view = view[view["region"].astype(str).isin(r_filter)]
    if p_filter:
        view = view[view["subsidy_program_name"].astype(str).isin(p_filter)]
    view = view[view["merit_score"] >= min_score]
    shortlist = view.sort_values("merit_score", ascending=False).head(top_n)

    st.subheader("Shortlist")
    display_cols = [
        "rank",
        "application_id",
        "region",
        "farm_district",
        "subsidy_program_name",
        "application_status",
        "merit_score",
        "eligible_amount_kzt",
    ]
    show = [c for c in display_cols if c in shortlist.columns]
    st.dataframe(shortlist[show], use_container_width=True, height=400)

    st.subheader("Объяснение по заявке (SHAP, топ-3)")
    idx_list = shortlist.index.tolist()
    if not idx_list:
        st.info("Нет строк после фильтров.")
        return

    pick = st.selectbox(
        "Выберите строку (индекс в полном наборе)",
        options=idx_list,
        format_func=lambda i: f"id={df.loc[i, 'application_id']} | {df.loc[i, 'region']}",
    )

    row_pos = df.index.get_loc(pick)
    reasons = top_feature_reasons(model, X, feature_cols, row_pos, top_k=3)
    for r in reasons:
        st.write(
            f"**{r['name']}** — значение `{r['feature_value']:.4g}`, "
            f"вклад SHAP `{r['shap_value']:.4g}` ({r['direction']})"
        )

    if ARTIFACTS_DIR.joinpath("training_meta.json").exists():
        with open(ARTIFACTS_DIR / "training_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        with st.expander("Метрики на hold-out и определение целевой переменной"):
            st.json(meta)


if __name__ == "__main__":
    main()
