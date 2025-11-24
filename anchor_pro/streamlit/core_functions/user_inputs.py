import streamlit as st
import numpy as np
from anchor_pro.elements.concrete_anchors import Profiles
from anchor_pro.elements.concrete_anchors import Profiles, AnchorPosition
import pandas as pd

def get_concrete_input_options():
    """
    Renders Streamlit widgets for selecting concrete properties.
    Returns a dictionary matching the fields of the ConcreteProps dataclass.
    """
    st.subheader("Concrete Properties")

    # Profile Selection mapped to Profiles Enum
    profile_label = st.selectbox(
        "Concrete Profile",
        options=[p.value for p in Profiles],
        index=0,  # Default to Slab
        help="Select the concrete member profile (e.g., Slab, Wall, Filled Deck).",
        key='concrete_profile'
    )
    # Convert label back to Enum member
    profile = next(p for p in Profiles if p.value == profile_label)

    # Weight Classification
    weight_classification = st.selectbox(
        "Weight Classification",
        options=["NWC", "LWC"],
        index=0, # Default to Normal Weight Concrete
        help="NWC: Normal Weight Concrete, LWC: Lightweight Concrete",
        key='weight_classification'
    )

    # Cracked Concrete
    cracked_concrete = st.checkbox(
        "Cracked Concrete",
        value=True,
        help="Assume concrete is cracked for calculation purposes (conservative).",
        key='cracked_concrete'
    )
    # Concrete Strength (fc)
    fc = st.number_input(
        "Compressive Strength (f'c) [psi]",
        min_value=2000.0,
        max_value=12000.0,
        value=4000.0,
        step=500.0,
        format="%.0f",
        key='concrete_strength',
    )

    # Lightweight Factor (lambda)
    # Default logic: 1.0 for NWC, 0.75 for LWC (user can adjust)
    if not 'lw_factor' in st.session_state:
        st.session_state['lw_factor'] = 1.0
    st.session_state['lw_factor'] = 1.0 if weight_classification == "NWC" else 0.75

    # Slab Thickness
    t_slab = st.number_input(
        "Member Thickness (h) [in]",
        min_value=0.0,
        value=12.0,
        step=0.5,
        help="Thickness of the concrete member.",
        key='slab_thickness'
    )

    # Poisson's Ratio (Advanced setting, usually hidden or defaulted)
    poisson = st.number_input(
        "Poisson's Ratio",
        min_value=0.1,
        max_value=0.3,
        value=0.2,
        step=0.05,
        key='poisson_ratio'
    )

def get_geo_input_options():
    """
    Renders Streamlit widgets for Anchor Geometry and Position.
    Returns a dictionary matching the fields of the GeoProps dataclass.
    """
    st.subheader("Geometry & Anchor Layout")

    # --- 1. Anchor Coordinates (Dynamic Table) ---
    st.markdown("##### Anchor Coordinates")
    
    # Initialize session state for anchors if it doesn't exist
    if 'geo_anchor_df' not in st.session_state:
        # Default 4-bolt pattern
        st.session_state['geo_anchor_df'] = pd.DataFrame({
            "x": [-3.0, 3.0, -3.0, 3.0],
            "y": [3.0, 3.0, -3.0, -3.0]
        })

    # Data Editor for adding/removing rows
    edited_df = st.data_editor(
        st.session_state['geo_anchor_df'],
        num_rows="dynamic",
        use_container_width=True,
        key='anchor_table_editor',
        column_config={
            "x": st.column_config.NumberColumn("x [in]", format="%.2f"),
            "y": st.column_config.NumberColumn("y [in]", format="%.2f"),
        }
    )
    # Update session state with edits so they persist
    st.session_state['geo_anchor_df'] = edited_df
    
    # Extract numpy array for GeoProps (n_anchor, 2)
    xy_anchors = edited_df.to_numpy()

    # --- 2. Fixture Dimensions (Bx, By) ---
    st.markdown("##### Fixture Dimensions")
    col_dims = st.columns(2)
    with col_dims[0]:
        Bx = st.number_input(
            "Fixture Width (Bx) [in]", 
            min_value=0.0, 
            value=10.0, 
            step=0.5,
            key="geo_Bx"
        )
    with col_dims[1]:
        By = st.number_input(
            "Fixture Height (By) [in]", 
            min_value=0.0, 
            value=10.0, 
            step=0.5,
            key="geo_By"
        )

    # --- 3. Edge Distances ---
    st.markdown("##### Concrete Edge Distances")
    st.caption("Distance from the fixture edge to the concrete edge. Uncheck 'Unbounded' to specify a distance.")

    # Helper function to generate edge inputs with Infinite toggle
    def render_edge_input(label, key_suffix):
        col_check, col_val = st.columns([0.4, 0.6])
        with col_check:
            # Checkbox: Is the edge unbounded (infinite)?
            is_inf = st.checkbox(f"Unbounded {label}", value=True, key=f"inf_{key_suffix}")
        with col_val:
            if is_inf:
                st.text_input(f"Dist. {label}", value="∞", disabled=True, label_visibility="collapsed", key=f"disp_{key_suffix}")
                return np.inf
            else:
                return st.number_input(
                    f"Distance {label}", 
                    min_value=0.0, 
                    value=12.0, 
                    step=1.0, 
                    label_visibility="collapsed", 
                    key=f"val_{key_suffix}"
                )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Left & Bottom**")
        cx_neg = render_edge_input("Left (-X)", "cx_neg")
        cy_neg = render_edge_input("Bottom (-Y)", "cy_neg")
    
    with c2:
        st.markdown("**Right & Top**")
        cx_pos = render_edge_input("Right (+X)", "cx_pos")
        cy_pos = render_edge_input("Top (+Y)", "cy_pos")

    # --- 4. Anchor Position ---
    st.markdown("##### Installation Position")
    pos_label = st.selectbox(
        "Anchor Position",
        options=[p.value for p in AnchorPosition],
        index=0,
        key="geo_anchor_pos"
    )
    # Convert label back to Enum
    anchor_position = next(p for p in AnchorPosition if p.value == pos_label)

    # Return dictionary matching GeoProps structure
    return {
        "xy_anchors": xy_anchors,
        "Bx": Bx,
        "By": By,
        "cx_neg": cx_neg,
        "cx_pos": cx_pos,
        "cy_neg": cy_neg,
        "cy_pos": cy_pos,
        "anchor_position": anchor_position
    }
