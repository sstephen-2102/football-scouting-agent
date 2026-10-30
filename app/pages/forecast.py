"""Member A — forecast lab."""

import streamlit as st

from app.components.results import artifact_gate, show_warnings


def render() -> None:
    st.title("Forecast Lab")
    st.caption(
        "Member A. Weekly Big-5 goals with STL, Holt-Winters, and ARIMA. "
        "Optional map into budget *B*."
    )
    ready = artifact_gate("forecast")
    if ready:
        st.write("Holdout MAPE and implied budget *B* will render here from `outputs/forecast/budget.json`.")
    else:
        st.write(
            "When `python -m src.pipeline forecast` has written `budget.json`, this page will show "
            "the holdout series and the implied transfer budget."
        )
    show_warnings(
        [
            "If the series is unstable, keep *B* as a user input on Squad Builder and still report the forecast.",
        ]
    )
