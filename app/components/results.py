"""Shared Streamlit chrome: metrics, not-ready gates, methodology notes."""

from numbers import Integral

import streamlit as st

from app.artifacts import is_ready, missing_message


def metric_row(values: dict, decimals: int = 3) -> None:
    for column, (label, value) in zip(st.columns(len(values)), values.items()):
        if isinstance(value, str):
            column.metric(label, value)
        elif isinstance(value, Integral):
            column.metric(label, f"{value:,}")
        else:
            column.metric(label, f"{value:.{decimals}f}")


def show_warnings(warnings: list[str], title: str = "Methodology notes") -> None:
    if warnings:
        st.info(f"**{title}**\n\n" + "\n".join(f"- {w}" for w in warnings))


def artifact_gate(name: str) -> bool:
    """Show the producer command when an export is missing. Return whether it is ready."""
    if is_ready(name):
        st.success(f"Artifact `{name}` is on disk.")
        return True
    st.error(missing_message(name))
    return False
