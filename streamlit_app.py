import streamlit as st
from anchor_pro.streamlit.core_ui.sidebar import render_sidebar

def main():
    st.write(st.session_state)
    render_sidebar()
    st.write(st.session_state)


if __name__ == "__main__":
    main()