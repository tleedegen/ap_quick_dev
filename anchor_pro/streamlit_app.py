import streamlit as st
from core_functions.visualizations import render_visualizations, render_project_designs_table, render_dynamic_design_table
from core_functions.save_project import render_save_load_section
from utils.session_state import app_setup, update_active_design, initialize_default_data_column
from utils.data_loader import anchor_pro_set_data
from auth.login_ui import render_login_sidebar
from auth.simple_auth import ensure_login
from core_ui.sidebar import render_sidebar
from core_ui.main_page import render_main_page



st.set_page_config(layout="wide")


def main():
    """Main function to run the AnchorPro app"""
    # Authentication check
    ensure_login()
    app_setup()
    # st.write(st.session_state)

    if "data_column" not in st.session_state:
        st.session_state['data_column'] = initialize_default_data_column()




    st.header("All Project Designs")
    # All inputs done in sidebar
    render_sidebar()
    # All data anchor calculations done
    anchor_pro_set_data()
    render_main_page()



    # st.write(st.session_state)


if __name__ == "__main__":
    main()
