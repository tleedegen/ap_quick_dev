import streamlit as st
from pandas import Series, DataFrame
from dataclasses import dataclass
from core_functions.design_parameters import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams
from utils.data_loader import anchor_pro_set_data, anchor_pro_concrete_data
from utils.exceptions import DataColumnError

def app_setup():
    """Initialize session state variables"""
    initialize_default_data_column()

    # Calcs and visualizations will be based on the active data column
    if 'active_data_column_index' not in st.session_state:
        st.session_state['active_data_column_index'] = 1

    if 'dynamic_data_column_index' not in st.session_state:
        st.session_state['dynamic_data_column_index'] = 0

    # Used for tracking latest changes between widgets and loaded data
    if 'global_version_counter' not in st.session_state:
        st.session_state['global_version_counter'] = 0
    if 'global_widget_version_counter' not in st.session_state:
        st.session_state['global_widget_version_counter'] = 0

def initialize_default_data_column() -> list[Series]:
    """Initialize the default data column in session state"""
    if "data_column" not in st.session_state:
        default_data_column: list = []
        design_params = DesignParameters()

        default_series = Series(design_params.combined_dict)
        default_data_column.insert(0, default_series)
        return default_data_column

def save_design_to_session(design_params: DesignParameters):
    """Save current design to session state as a new snapshot (list[Series], newest first)."""

    # ensure data_column is a list of Series (recover if it was a single Series)
    existing_data_column = st.session_state['data_column']
    if isinstance(existing_data_column, list) and len(existing_data_column) > 0 and all(isinstance(item, Series) for item in existing_data_column):
        data_list = existing_data_column
    else:
        raise DataColumnError("Invalid data column format")


    # take an independent snapshot so previous saves don't get mutated later
    snap = Series(design_params.combined_dict).copy(deep=True)
    data_list.append(snap)

    st.session_state["data_column"] = data_list

def add_design_names_to_session(name: str):
    """Add design names to session state"""
    if "design_names" not in st.session_state:
        st.session_state["design_names"] = []

    st.session_state["design_names"].append(name)

def overwrite_design_names_to_session(name: str, index: int):
    """Overwrite design name in session state"""
    if 'design_names' not in st.session_state:
        st.error("No design names found in session state.")
    elif len(st.session_state['design_names']) > 1 :
        st.session_state['design_names'][index - 1] = name


def get_saved_designs():
    """Get all saved designs from session state"""
    return st.session_state.get("data_column")

def update_active_data_column(data_column_key: str, data):
    """Update precise session state data column value"""
    if st.session_state['data_column']:
        st.session_state['data_column'][0][data_column_key] = data


def update_active_design(design_params: DesignParameters):
    """Update the active design in session state"""
    if "data_column" not in st.session_state or not isinstance(st.session_state["data_column"], list):
        st.session_state["data_column"] = []

    st.session_state['data_column'].insert(0, Series(design_params.combined_dict))

    if len(st.session_state['data_column']) > 1:
        st.session_state['data_column'].pop(1)


