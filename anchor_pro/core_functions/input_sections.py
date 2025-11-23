import streamlit as st
from core_functions.design_parameters import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams, Anchor, BasePlate
from utils.data_loader import  get_anchor_products
import pandas as pd
from utils.widget_generator import WidgetSpecs
from utils.constants import ANCHOR_PRO_BACKEND_PARAM_SS_KEYS


def render_substrate_section() -> SubstrateParams:
    """Render substrate input fields and records data to session_state data_column"""
    with st.expander("Substrate Parameters", expanded=True):

        st.subheader("Substrate")
        substrate_params = SubstrateParams()
        base_material_widget = WidgetSpecs(
            label=substrate_params.Fields.BaseMaterial.label,
            param_type=substrate_params,
            widget_type='selectbox'
            )

        cracked_concrete_widget = WidgetSpecs(
            label=substrate_params.Fields.CrackedConcrete.label,
            param_type=substrate_params,
            widget_type='selectbox'
        )

        grouted_widget = WidgetSpecs(
            label=substrate_params.Fields.Grouted.label,
            param_type=substrate_params,
            widget_type='selectbox'
        )

        weight_class_widget = WidgetSpecs(
            label=substrate_params.Fields.WeightClass.label,
            param_type=substrate_params,
            widget_type='selectbox'
        )

        poisson_widget = WidgetSpecs(
            label=substrate_params.Fields.Poisson.label,
            param_type=substrate_params,
            widget_type='number_input'
        )


        concrete_thickness_widget = WidgetSpecs(
            label=substrate_params.Fields.ConcreteThickness.label,
            param_type=substrate_params,
            widget_type='number_input'
        )
        profile_widget = WidgetSpecs(
            label=substrate_params.Fields.Profile.label,
            param_type=substrate_params,
            widget_type='selectbox'
        )
            

        edge_dist_x_neg_widget = WidgetSpecs(
            label=substrate_params.Fields.EdgeDistXNeg.label,
            param_type=substrate_params,
            widget_type='number_input'
        )

        edge_dist_x_pos_widget = WidgetSpecs(
            label=substrate_params.Fields.EdgeDistXPos.label,
            param_type=substrate_params,
            widget_type='number_input'
        )
        

        edge_dist_y_neg_widget = WidgetSpecs(
            label=substrate_params.Fields.EdgeDistYNeg.label,
            param_type=substrate_params,
            widget_type='number_input'
        )
        

        edge_dist_y_pos_widget = WidgetSpecs(
            label=substrate_params.Fields.EdgeDistYPos.label,
            param_type=substrate_params,
            widget_type='number_input'
        )
        
        anchor_position_widget = WidgetSpecs(
            label=substrate_params.Fields.AnchorPosition.label,
            param_type=substrate_params,
            widget_type='selectbox'
        )

        deck_location_widget = WidgetSpecs(
            label=substrate_params.Fields.DeckLocation.label,
            param_type=substrate_params,
            widget_type='selectbox'
        )

        hole_diameter_widget = WidgetSpecs(
            label=substrate_params.Fields.HoleDiameter.label,
            param_type=substrate_params,
            widget_type='number_input'
        )

        face_side_widget = WidgetSpecs(
            label=substrate_params.Fields.FaceSide.label,
            param_type=substrate_params,
            widget_type='selectbox'
        )
        
    substrate_params = SubstrateParams(
        fc=st.session_state['data_column'][0]['fc'],
        cracked_concrete=st.session_state['data_column'][0]['cracked_concrete'],
        grouted=st.session_state['data_column'][0]['grouted'],
        lw_factor=st.session_state['data_column'][0]['lw_factor'],
        poisson=st.session_state['data_column'][0]['poisson'],
        weight_classification_base=st.session_state['data_column'][0]['weight_classification_base'],
        t_slab=st.session_state['data_column'][0]['t_slab'],
        profile=st.session_state['data_column'][0]['profile'],
        cx_neg=st.session_state['data_column'][0]['cx_neg'],
        cx_pos=st.session_state['data_column'][0]['cx_pos'],
        cy_neg=st.session_state['data_column'][0]['cy_neg'],
        cy_pos=st.session_state['data_column'][0]['cy_pos'],
        anchor_position=st.session_state['data_column'][0]['anchor_position'],
        deck_location=st.session_state['data_column'][0]['deck_location'],
        hole_diameter=st.session_state['data_column'][0]['hole_diameter'],
        face_side=st.session_state['data_column'][0]['face_side']
    )
    return substrate_params

def render_anchor_product_section() -> AnchorProduct:
    """Render anchor product selection fields and records data to sessions_state data_column"""
    with st.expander("Anchor Product", expanded=True):
        st.subheader("Anchor Product")

        anchor_product = AnchorProduct()

        # Manufacturer widget
        manufacturer_widget = WidgetSpecs(
            label=anchor_product.Fields.Manufacturer.label,
            param_type=anchor_product,
            widget_type='selectbox'
        )

        # Get the manufacturer value from session state
        manufacturer = st.session_state['manufacturer']

        # Filter products based on manufacturer if selected
        if manufacturer:
            anchor_products = get_anchor_products(anchor_product.anchor_parameters,
                                                manufacturer=manufacturer)
        else:
            anchor_products = get_anchor_products(anchor_product.anchor_parameters)

        # Update options for specified product dynamically
        anchor_product.Fields.SpecifiedProduct.options = list(anchor_products)

        # Specified product widget
        specified_product_widget = WidgetSpecs(
            label=anchor_product.Fields.SpecifiedProduct.label,
            param_type=anchor_product,
            widget_type='selectbox',
            options=list(anchor_products)  # Pass the filtered options
        )

        anchor_product = AnchorProduct(
            specified_product=st.session_state['data_column'][0]['specified_product'],
        )
        return anchor_product

def render_anchor_loading_section() -> LoadingParams:
    """Render anchor loading input fields"""
    with st.expander("Anchor Loading", expanded=True):
        st.subheader("Anchor Loading")

        loading_params = LoadingParams()
        
        load_location_widget = WidgetSpecs(
            label=loading_params.Fields.LoadLocation.label,
            param_type=loading_params,
            widget_type='selectbox'
        )

        st.markdown("**Options:**")
        col1, col2 = st.columns(2)
        
        with col1:
            seismic_widget = WidgetSpecs(
                label=loading_params.Fields.Seismic.label,
                param_type=loading_params,
                widget_type='selectbox'
            )
        
        with col2:
            phi_override_widget = WidgetSpecs(
                label=loading_params.Fields.PhiOverride.label,
                param_type=loading_params,
                widget_type='selectbox'
            )
            
    loading_params = LoadingParams(
        location=st.session_state['data_column'][0]['location'],
        seismic=st.session_state['data_column'][0]['seismic'],
        phi_override=st.session_state['data_column'][0]['phi_override']
    )
    return loading_params

def render_installation_section() -> InstallationParams:

    """Render installation conditions fields"""
    with st.expander('Installation Conditions', expanded=True):
        st.header("Installation Conditions")
        
        # First row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hef = st.number_input(
                "hef (in)",
                min_value=0.0,
                value=None,
                placeholder="Input...",
                key="hef"
            )
            
            drilling_type = st.selectbox(
                "Drilling Type",
                options=[
                    None,
                    "Hammer Drill",
                    "Core Drill",
                    "Diamond Core",
                    "Rotary Impact"
                ],
                index=0,
                placeholder="Select...",
                key="drilling_type"
            )
        
        with col2:
            short_term_temp = st.number_input(
                "Short Term Temp (°F)",
                value=None,
                placeholder="Input...",
                key="short_term_temp"
            )
            
            inspection_condition = st.selectbox(
                "Inspection Condition",
                options=[
                    None,
                    "Continuous",
                    "Periodic",
                    "None"
                ],
                index=0,
                placeholder="Select...",
                key="inspection_condition"
            )
        
        with col3:
            long_term_temp = st.number_input(
                "Long Term Temp (°F)",
                value=None,
                placeholder="Input...",
                key="long_term_temp"
            )
            
            moisture_condition = st.selectbox(
                "Moisture Condition",
                options=[
                    None,
                    "Dry",
                    "Water-Saturated",
                    "Water-Filled",
                    "Submerged"
                ],
                index=0,
                placeholder="Select...",
                key="moisture_condition"
            )
        
        return InstallationParams(
            hef=hef,
            short_term_temp=short_term_temp,
            long_term_temp=long_term_temp,
            drilling_type=drilling_type,
            inspection_condition=inspection_condition,
            moisture_condition=moisture_condition,
        )

def load_design_to_editor(design: pd.Series):
    '''Load selected design values into the widgets for editing'''
