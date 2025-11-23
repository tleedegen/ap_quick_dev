import streamlit as st
from core_functions.visualizations import render_visualizations, render_project_designs_table, render_dynamic_design_table, render_demand_capacity_ratio_table
from core_functions.save_project import render_save_load_section
from utils.session_state import app_setup, update_active_design, initialize_default_data_column
from utils.data_loader import anchor_pro_set_data
from auth.login_ui import render_login_sidebar
from auth.simple_auth import ensure_login
from core_ui.sidebar import render_sidebar


def render_main_page():
    """Render the main page of the AnchorPro app"""
    st.image('https://degenkolb.com/wp-content/uploads/Degenkolb-wh-logo.svg')
    st.title("AnchorPro Concrete")

    render_dynamic_design_table()
    render_project_designs_table()
    render_demand_capacity_ratio_table()

    render_save_load_section()
    st.markdown("---")
    render_visualizations()