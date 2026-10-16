"""A local report viewer for recorded experiments. It is not a finished case product."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Datathon · Experiments", page_icon="◉", layout="wide")
st.title("Результаты экспериментов")
runs = sorted(Path("reports/runs").glob("*/metrics.json"), reverse=True)
if not runs:
    st.info("Сначала запустите datathon train --config configs/synthetic.yaml")
    st.stop()
choice = st.selectbox("Эксперимент", runs, format_func=lambda p: p.parent.name)
result = json.loads(choice.read_text())
run = choice.parent
cfg_text = (run / "config.yaml").read_text()
if "data/synthetic" in cfg_text:
    st.warning("Учебные синтетические данные. Эти результаты не относятся к кейсу ForteBank.")
st.caption(f"Разбиение: {result['split_strategy']} · Размеры: {result['partition_sizes']}")
scores = pd.DataFrame({"Dummy": result["baseline_holdout"], "CatBoost": result["holdout"]})
st.dataframe(scores, width="stretch")
st.image(str(run / "feature_importance.png"))
st.caption("Важность признаков не доказывает причинность. Метрики рассчитаны на одном holdout.")
st.download_button("Скачать метрики", (run / "metrics.json").read_bytes(), "metrics.json")
with st.expander("Условия эксперимента"):
    st.code(cfg_text, language="yaml")
