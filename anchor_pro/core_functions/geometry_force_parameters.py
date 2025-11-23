import streamlit as st
import pandas as pd
from core_functions.design_parameters import Anchor, BasePlate
from utils.widget_generator import WidgetSpecs


def render_anchor_data_editor() -> Anchor:
    """Render the anchor geometry and loads editor"""

    with st.expander("Anchor Geometry & Loads", expanded=True):
        st.subheader("Anchor Geometry & Forces")

        anchor_table = st.empty()
        anchor_buttons = st.empty()

        # Initialize with empty dataframe if not exists
        if "anchor_data" not in st.session_state:
            st.session_state.anchor_data = pd.DataFrame({
                'X': [0.0, 6.0],
                'Y': [0.0, 6.0],
                'Vx': [0.0, 0.0],
                'Vy': [0.0, 0.0],
                'N': [1000, 1000]
            })

        with anchor_buttons.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Add Anchor", key="btn_add_anchor"):
                    new_anchor = pd.DataFrame({
                        'X': [0.0],
                        'Y': [0.0],
                        'Vx': [0.0],
                        'Vy': [0.0],
                        'N': [0.0]
                    })
                    st.session_state['data_column'][0]['anchor_geometry_forces'] = pd.concat(
                        [st.session_state['data_column'][0]['anchor_geometry_forces'], new_anchor], ignore_index=True
                    )
                    st.rerun()
            with col2:
                if st.button('Remove Anchor', key="btn_remove_anchor"):
                    if not st.session_state['data_column'][0]['anchor_geometry_forces'].empty:
                        st.session_state['data_column'][0]['anchor_geometry_forces'] = st.session_state['data_column'][0]['anchor_geometry_forces'].iloc[:-1]
                    st.rerun()
            with col3:
                if st.button("Clear All Anchors", key="btn_clear_anchors"):
                    st.session_state['data_column'][0]['anchor_geometry_forces'] = pd.DataFrame({
                        'X': [0.0],
                        'Y': [0.0],
                        'Vx': [0.0],
                        'Vy': [0.0],
                        'N': [0.0]
                    })
                    st.rerun()
    # Create column configuration for better data entry
    column_config = {
        "X": st.column_config.NumberColumn(
            "X (in)",
            help="X-coordinate of anchor",
            min_value=None,
            max_value=None,
        ),
        "Y": st.column_config.NumberColumn(
            "Y (in)", 
            help="Y-coordinate of anchor",
            min_value=None,
            max_value=None,
        ),
        "Vx": st.column_config.NumberColumn(
            "Vx (lbs)",
            help="Shear force in X direction",
            min_value=None,
            max_value=None
        ),
        "Vy": st.column_config.NumberColumn(
            "Vy (lbs)",
            help="Shear force in Y direction",
            min_value=None,
            max_value=None
        ),
        "N": st.column_config.NumberColumn(
            "N (lbs)",
            help="Tension force (positive = tension)",
            min_value=None,
            max_value=None,
            step=1.0,
        )
    }
    # Create the data editor - use the return value directly

    current_df = st.session_state['data_column'][0]['anchor_geometry_forces']

    edited_df = anchor_table.data_editor(
        data=current_df.copy(),
        column_config=column_config,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        key="anchor_editor"
    )
    
    if not edited_df.equals(current_df):
        st.session_state['data_column'][0]['anchor_geometry_forces'] = edited_df
        st.rerun()
    #TODO: Use backend logic for min_spacing checks
    # # Add validation warnings
    # if len(anchor_geometry_df) > 1:
    #     # Check for minimum spacing
    #     min_spacing = check_minimum_spacing(anchor_geometry_df)
    #     if min_spacing < 3.0:  # Typical minimum spacing requirement
    #         st.warning(f"⚠️ Minimum anchor spacing is {min_spacing:.2f} inches. Consider minimum 3.0 inches.")
    
    # Quick action buttons - Streamlit will automatically rerun when session state changes
    
    anchor = Anchor(anchor_geometry_forces=st.session_state['data_column'][0]['anchor_geometry_forces'])
    return anchor


def render_baseplate_geometry() -> BasePlate:
    """Render baseplate geometry editor using WidgetSpecs"""
    with st.expander("Baseplate Geometry", expanded=True):
        st.subheader('Baseplate Geometry & Forces')
        
        baseplate = BasePlate()

        # Base plate width widget
        bx_widget = WidgetSpecs(
            label=baseplate.Fields.Bx.label,
            param_type=baseplate,
            widget_type='number_input'
        )

        # Base plate length widget
        by_widget = WidgetSpecs(
            label=baseplate.Fields.By.label,
            param_type=baseplate,
            widget_type='number_input'
        )
        
        st.markdown("**Moments:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mx_widget = WidgetSpecs(
                label=baseplate.Fields.Mx.label,
                param_type=baseplate,
                widget_type='number_input'
            )
        
        with col2:
            my_widget = WidgetSpecs(
                label=baseplate.Fields.My.label,
                param_type=baseplate,
                widget_type='number_input'
            )
        
        with col3:
            mz_widget = WidgetSpecs(
                label=baseplate.Fields.Mz.label,
                param_type=baseplate,
                widget_type='number_input'
            )
    
    baseplate = BasePlate(
        Bx=st.session_state['data_column'][0]['Bx'],
        By=st.session_state['data_column'][0]['By'],
        mx=st.session_state['data_column'][0]['mx'],
        my=st.session_state['data_column'][0]['my'],
        mz=st.session_state['data_column'][0]['mz']
    )
    return baseplate