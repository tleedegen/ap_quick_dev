import streamlit as st
from anchor_pro.streamlit.core_functions.user_inputs import get_concrete_input_options

def render_sidebar():
    with st.sidebar:
        st.header("Design Editor")
        concrete_inputs = get_concrete_input_options()