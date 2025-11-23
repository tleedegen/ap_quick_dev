"""Module for saving and loading project designs"""

import streamlit as st
import pickle
import io
from typing import List, Optional
from pandas import Series
from datetime import datetime


def save_project() -> Optional[bytes]:
    """
    Serialize the current project data from session state.
    Returns the pickled bytes or None if no data exists.
    """
    if 'data_column' not in st.session_state or not st.session_state.data_column:
        return None

    try:
        # Serialize the data_column list using pickle
        return pickle.dumps(st.session_state.data_column)
    except Exception as e:
        st.error(f"Error serializing project data: {str(e)}")
        return None

def load_project(file_bytes: bytes) -> bool:
    """
    Deserialize and load project data into session state.
    Returns True if successful, False otherwise.
    """
    try:
        # Deserialize the data
        data_column = pickle.loads(file_bytes)

        # Validate that it's a list of Series
        if not isinstance(data_column, list):
            st.error("Invalid project file: Expected a list structure")
            return False

        # Check if all elements are pandas Series
        for item in data_column:
            if not isinstance(item, Series):
                st.error("Invalid project file: Expected pandas Series objects")
                return False

        # Update session state
        st.session_state.data_column = data_column
        st.session_state.data_column_counter = len(data_column)

        # bump the project epoch so widgets do a one-time model → UI sync
        st.session_state['global_version_counter'] += 1
        st.session_state['project_loaded_at'] = datetime.now().isoformat()
        return True

    except pickle.UnpicklingError:
        st.error("Invalid project file: Could not unpickle the data")
        return False
    except Exception as e:
        st.error(f"Error loading project: {str(e)}")
        return False


def render_save_load_ui():
    """
    Render the save/load UI components in the Streamlit app.
    This function creates the save and upload buttons.
    """

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    with col1:
        # Save Project button
        if st.button("💾 Save Project", type="secondary", use_container_width=True):
            project_data = save_project()

            if project_data:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"anchorpro_project_{timestamp}.pkl"

                # Create download button
                st.download_button(
                    label="📥 Download Project File",
                    data=project_data,
                    file_name=filename,
                    mime="application/octet-stream",
                    use_container_width=True
                )
                st.success("Project ready for download!")
            else:
                st.warning("No project data to save. Please add at least one design.")

    with col2:
        # Upload Project button with unique key
        uploaded_file = st.file_uploader(
            "📤 Load Project",
            type=['pkl'],
            help="Upload a previously saved AnchorPro project file",
            label_visibility="collapsed",
            key="project_uploader"
        )

        # Initialize tracking in session state
        if 'last_processed_file' not in st.session_state:
            st.session_state.last_processed_file = None

        if uploaded_file is not None:
            # Create a unique identifier for the file
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"

            # Only process if this is a new file
            if st.session_state.last_processed_file != file_id:
                # Read the file bytes
                file_bytes = uploaded_file.read()

                # Load the project
                if load_project(file_bytes):
                    st.success("Project loaded successfully!")
                    st.session_state.last_processed_file = file_id
                    st.rerun()  # Now safe to rerun
                else:
                    st.error("Failed to load project file")
                    st.session_state.last_processed_file = None

def render_save_load_section():
    """
    Render a dedicated section for save/load functionality.
    This can be called from the main app or placed in an expander.
    """
    with st.expander("💾 Save/Load Project", expanded=True):
        st.markdown("""
        **Save your work** to continue later or share with colleagues.
        - **Save Project**: Downloads your current designs as a file
        - **Load Project**: Upload a previously saved project file
        """)

        render_save_load_ui()

        # Show current project status
        if 'data_column' in st.session_state and st.session_state.data_column:
            num_designs = len(st.session_state.data_column) - 1  # Exclude the default design
            st.info(f"Current project has {num_designs} saved design{'s' if num_designs != 1 else ''}")
        else:
            st.info("No designs in current project")


def quick_save_load_buttons():
    """
    Minimal UI for save/load - just the buttons without extra text.
    Can be placed in sidebar or main area.
    """
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save", key="quick_save", use_container_width=True):
            project_data = save_project()
            if project_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "⬇ Download",
                    data=project_data,
                    file_name=f"anchorpro_{timestamp}.pkl",
                    mime="application/octet-stream",
                    key="quick_download",
                    use_container_width=True
                )

    with col2:
        uploaded = st.file_uploader(
            "Load",
            type=['pkl'],
            key="quick_upload",
            label_visibility="visible"
        )

        # Track processed files for this uploader too
        if 'last_processed_quick_file' not in st.session_state:
            st.session_state.last_processed_quick_file = None

        if uploaded:
            file_id = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.last_processed_quick_file != file_id:
                if load_project(uploaded.read()):
                    st.session_state.last_processed_quick_file = file_id
                    st.rerun()


# Optional: Add validation function for integrity checks
def validate_project_data(data_column: List[Series]) -> tuple[bool, str]:
    """
    Validate the structure and content of project data.
    Returns (is_valid, message).
    """
    if not data_column:
        return False, "Project contains no designs"

    required_keys = ['fc', 'Bx', 'By']  # Add more as needed

    for i, series in enumerate(data_column):
        for key in required_keys:
            if key not in series.index:
                return False, f"Design {i+1} missing required field: {key}"

        # Check for anchor geometry DataFrame if it should exist
        if 'anchor_geometry_forces' in series.index:
            import pandas as pd
            if not isinstance(series['anchor_geometry_forces'], pd.DataFrame):
                return False, f"Design {i+1} has invalid anchor geometry data"

    return True, "Project data is valid"


# Alternative approach using a callback pattern (optional, more robust)
def load_project_with_callback():
    """
    Alternative implementation using Streamlit's callback system.
    This approach prevents the infinite loop more reliably.
    """
    def _load_callback():
        """Callback function to process the uploaded file"""
        if st.session_state.project_file_upload is not None:
            file_bytes = st.session_state.project_file_upload.read()
            if load_project(file_bytes):
                st.success("Project loaded successfully!")
                # Clear the uploader by setting a flag
                st.session_state.should_clear_uploader = True

    # Check if we should clear the uploader
    if st.session_state.get('should_clear_uploader', False):
        st.session_state.should_clear_uploader = False
        st.session_state.project_file_upload = None

    # File uploader with callback
    st.file_uploader(
        "📤 Load Project (Alternative)",
        type=['pkl'],
        key="project_file_upload",
        on_change=_load_callback,
        help="Upload a previously saved AnchorPro project file"
    )


# Export main functions for use in other modules
__all__ = [
    'save_project',
    'load_project',
    'render_save_load_ui',
    'render_save_load_section',
    'quick_save_load_buttons',
    'validate_project_data',
    'load_project_with_callback'
]