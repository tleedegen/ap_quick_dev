import streamlit as st
import numpy as np
from anchor_pro.elements.concrete_anchors import Profiles

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
