"""Navigation shell. Add page modules as each member lands exports."""

import streamlit as st

from app.pages import overview


def main() -> None:
    st.set_page_config(page_title="Football Scouting Agent", layout="wide")
    pages = {
        "Overview": st.Page(overview.render, title="Overview", default=True),
    }
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
