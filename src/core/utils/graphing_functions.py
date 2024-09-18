# -*- encoding: utf-8 -*-
"""
Graphing functions to plot
"""

import streamlit as st

__all__ = [
    "local_css",
]


# CSS Styling
def local_css(file_path: str):
    """
    Reads css file and applies the styling to the streamlit page.

    Parameters
    ----------
    file_path : str
        Relative path to the css file.

    Returns
    -------
    None.

    """
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
