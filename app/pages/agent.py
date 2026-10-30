"""Module 1 — PEAS and ten environment dimensions."""

import pandas as pd
import streamlit as st

from src.agent.peas import DISCLAIMER, PEAS, ten_environment_dimensions


def render() -> None:
    st.title("Agent (PEAS)")
    st.caption("Module 1 framing. Performance, environment, actuators, sensors, and the ten environment dimensions.")
    items = list(PEAS.items())
    for start in (0, 2):
        columns = st.columns(2)
        for column, (key, value) in zip(columns, items[start : start + 2]):
            column.subheader(key.capitalize())
            column.write(value)
    st.subheader("Ten environment dimensions")
    table = pd.DataFrame(
        [{"axis": axis, "this project": text} for axis, text in ten_environment_dimensions().items()]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.info(DISCLAIMER)
