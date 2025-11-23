import streamlit as st
from auth.login_ui import render_login_sidebar
from core_functions.design_parameters import DesignParameters, SubstrateParams, AnchorProduct, LoadingParams, InstallationParams, Anchor, BasePlate
from utils.session_state import save_design_to_session, add_design_names_to_session, overwrite_design_names_to_session
from core_functions.input_sections import render_substrate_section, render_anchor_product_section, render_anchor_loading_section, render_installation_section
from core_functions.geometry_force_parameters import render_anchor_data_editor, render_baseplate_geometry
from utils.widget_generator import copy_design_params_to_widgets

def render_sidebar():
    """Render the sidebar with all input fields"""
    with st.sidebar:
        st.header("Design Editor")

        # Initialize design parameters
        substrate_params: SubstrateParams = render_substrate_section()
        anchor_product: AnchorProduct = render_anchor_product_section()
        loading_params: LoadingParams = render_anchor_loading_section()
        anchor: Anchor = render_anchor_data_editor()
        baseplate: BasePlate = render_baseplate_geometry()
        install_params: InstallationParams = render_installation_section()

        design_params: DesignParameters = DesignParameters(
            substrate=substrate_params,
            anchor_product=anchor_product,
            loading=loading_params,
            anchor=anchor,
            baseplate=baseplate,
            installation=install_params
        )


        render_copy_design_button()
        render_overwrite_button()


        if st.button("Save Data As...", type="primary", use_container_width=True):
            design_name_dialog(design_params)

    render_login_sidebar()

@st.dialog('Save Design')
def design_name_dialog(design_params: DesignParameters):
    name = st.text_input('Enter name for this design')
    if st.button('Save'):
        add_design_names_to_session(name)
        st.success(f"Design saved as '{name}'")
        save_design_to_session(design_params)

        st.success("Design saved!")
        st.rerun()

def render_overwrite_button():
    @st.dialog('Overwrite Design')
    def overwrite_design():
        design_index = st.selectbox(label='Select Design to Overwrite', options=range(1,len(st.session_state['data_column'])), format_func=lambda x: st.session_state['design_names'][x-1])
        name = st.text_input(label='Enter name for this design')
        to_overwrite = st.session_state['design_names'][design_index-1]
        if st.button(label=f'Overwrite {to_overwrite} with {name}', type='secondary', use_container_width=True):
            overwrite_design_names_to_session(name, design_index)
            st.rerun()
    if st.button('Overwrite Current Design', type='secondary', use_container_width=True, key='btn_coverwrite_design'):
        overwrite_design()

def render_copy_design_button():
    @st.dialog('Copy Design Values')
    def copy_design_to_widgets():
        design_index = st.selectbox(label='Select Design to Overwrite', options=range(1,len(st.session_state['data_column'])), format_func=lambda x: st.session_state['design_names'][x-1])

        if st.button(label='Copy Selected Design to Widgets', type='secondary', key='btn_copy_design'):
            copy_design_params_to_widgets(design_index)
            st.rerun()
    if st.button('Copy Design to Editor', type='secondary', use_container_width=True):
        copy_design_to_widgets()

