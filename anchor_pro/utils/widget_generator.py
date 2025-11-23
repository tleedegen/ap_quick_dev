from dataclasses import dataclass
from typing import Any, Optional, ClassVar, Callable, Iterable
import streamlit as st
from core_functions.design_parameters import SubstrateParams, AnchorProduct, LoadingParams, InstallationParams, Params, BasePlate
from utils.constants import ANCHOR_PRO_BACKEND_PARAM_SS_KEYS
from utils.exceptions import DataColumnError

@dataclass
class WidgetSpecs:
    option_index: Optional[int] = None
    widget_type: Optional[str] = None
    key: Optional[str] = None
    label: Optional[str] = None
    options: Optional[tuple] = None
    value: Optional[Any] = None
    param_type: Optional[Params] = None
    on_change: Optional[Callable] = None
    visible: bool = True
    placeholder: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    format_func: Optional[Callable] = None
    index: Optional[int] = None

    WIDGET_TYPES = {
        'selectbox': st.selectbox,
        'number_input': st.number_input,
        'data_editor': st.data_editor
    }

    def __post_init__(self):
        dynamic_index: int = 0

        # SubstrateParams widgets
        if isinstance(self.param_type, SubstrateParams):
            if self.label == SubstrateParams.Fields.BaseMaterial.label:
                self.key = self.param_type.Fields.BaseMaterial.key
                self.options = self.param_type.Fields.BaseMaterial.options
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['fc']
                    self.value = st.session_state['data_column'][dynamic_index]['fc']

            elif self.label == SubstrateParams.Fields.CrackedConcrete.label:
                self.key = self.param_type.Fields.CrackedConcrete.key
                self.options = self.param_type.Fields.CrackedConcrete.options
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['cracked_concrete']
                    self.value = st.session_state['data_column'][dynamic_index]['cracked_concrete']

            elif self.label == SubstrateParams.Fields.Grouted.label:
                self.key = self.param_type.Fields.Grouted.key
                self.options = self.param_type.Fields.Grouted.options
                self.placeholder = self.param_type.Fields.Grouted.placeholder
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['grouted']

            elif self.label == SubstrateParams.Fields.WeightClass.label:
                self.key = self.param_type.Fields.WeightClass.key
                self.options = self.param_type.Fields.WeightClass.options
                self.index = self.param_type.Fields.WeightClass.index
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['weight_classification_base']

            elif self.label == SubstrateParams.Fields.Poisson.label:
                self.key = self.param_type.Fields.Poisson.key
                self.min_value = self.param_type.Fields.Poisson.min_value
                self.max_value = self.param_type.Fields.Poisson.max_value
                self.value = self.param_type.Fields.Poisson.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['poisson']

            elif self.label == SubstrateParams.Fields.ConcreteThickness.label:
                self.key = self.param_type.Fields.ConcreteThickness.key
                self.min_value = self.param_type.Fields.ConcreteThickness.min_value
                self.value = self.param_type.Fields.ConcreteThickness.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['t_slab']

            elif self.label == SubstrateParams.Fields.Profile.label:
                self.key = self.param_type.Fields.Profile.key
                self.options = self.param_type.Fields.Profile.options
                self.index = self.param_type.Fields.Profile.index
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['profile']

            elif self.label == SubstrateParams.Fields.EdgeDistXNeg.label:
                self.key = self.param_type.Fields.EdgeDistXNeg.key
                self.min_value = self.param_type.Fields.EdgeDistXNeg.min_value
                self.value = self.param_type.Fields.EdgeDistXNeg.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['cx_neg']

            elif self.label == SubstrateParams.Fields.EdgeDistXPos.label:
                self.key = self.param_type.Fields.EdgeDistXPos.key
                self.min_value = self.param_type.Fields.EdgeDistXPos.min_value
                self.value = self.param_type.Fields.EdgeDistXPos.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['cx_pos']

            elif self.label == SubstrateParams.Fields.EdgeDistYNeg.label:
                self.key = self.param_type.Fields.EdgeDistYNeg.key
                self.min_value = self.param_type.Fields.EdgeDistYNeg.min_value
                self.value = self.param_type.Fields.EdgeDistYNeg.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['cy_neg']

            elif self.label == SubstrateParams.Fields.EdgeDistYPos.label:
                self.key = self.param_type.Fields.EdgeDistYPos.key
                self.min_value = self.param_type.Fields.EdgeDistYPos.min_value
                self.value = self.param_type.Fields.EdgeDistYPos.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['cy_pos']

            elif self.label == SubstrateParams.Fields.AnchorPosition.label:
                self.key = self.param_type.Fields.AnchorPosition.key
                self.options = self.param_type.Fields.AnchorPosition.options
                self.placeholder = self.param_type.Fields.AnchorPosition.placeholder
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['anchor_position']

            elif self.label == SubstrateParams.Fields.DeckLocation.label:
                self.key = self.param_type.Fields.DeckLocation.key
                self.options = self.param_type.Fields.DeckLocation.options
                self.placeholder = self.param_type.Fields.DeckLocation.placeholder
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['deck_location']

            elif self.label == SubstrateParams.Fields.HoleDiameter.label:
                self.key = self.param_type.Fields.HoleDiameter.key
                self.min_value = self.param_type.Fields.HoleDiameter.min_value
                self.value = self.param_type.Fields.HoleDiameter.value
                self.placeholder = self.param_type.Fields.HoleDiameter.placeholder
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['hole_diameter']

            elif self.label == SubstrateParams.Fields.FaceSide.label:
                self.key = self.param_type.Fields.FaceSide.key
                self.options = self.param_type.Fields.FaceSide.options
                self.placeholder = self.param_type.Fields.FaceSide.placeholder
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['face_side']

        # AnchorProduct widgets
        elif isinstance(self.param_type, AnchorProduct):
            if self.label == AnchorProduct.Fields.Manufacturer.label:
                self.key = self.param_type.Fields.Manufacturer.key
                self.options = self.param_type.Fields.Manufacturer.options
                self.placeholder = self.param_type.Fields.Manufacturer.placeholder
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index].get('mode')

            elif self.label == AnchorProduct.Fields.SpecifiedProduct.label:
                self.key = self.param_type.Fields.SpecifiedProduct.key
                self.placeholder = self.param_type.Fields.SpecifiedProduct.placeholder
                self.index = self.param_type.Fields.SpecifiedProduct.index
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['specified_product']

        # LoadingParams widgets
        elif isinstance(self.param_type, LoadingParams):
            if self.label == LoadingParams.Fields.LoadLocation.label:
                self.key = self.param_type.Fields.LoadLocation.key
                self.options = self.param_type.Fields.LoadLocation.options
                self.index = self.param_type.Fields.LoadLocation.index
                self.placeholder = self.param_type.Fields.LoadLocation.placeholder
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['location']

            elif self.label == LoadingParams.Fields.Seismic.label:
                self.key = self.param_type.Fields.Seismic.key
                self.options = self.param_type.Fields.Seismic.options
                self.index = self.param_type.Fields.Seismic.index
                self.format_func = lambda x: "Yes" if x else "No"
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['seismic']

            elif self.label == LoadingParams.Fields.PhiOverride.label:
                self.key = self.param_type.Fields.PhiOverride.key
                self.options = self.param_type.Fields.PhiOverride.options
                self.index = self.param_type.Fields.PhiOverride.index
                self.format_func = lambda x: "Yes" if x else "No"
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['phi_override']

        # BasePlate widgets
        elif isinstance(self.param_type, BasePlate):
            if self.label == BasePlate.Fields.Bx.label:
                self.key = self.param_type.Fields.Bx.key
                self.min_value = self.param_type.Fields.Bx.min_value
                self.value = self.param_type.Fields.Bx.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['Bx']
            elif self.label == BasePlate.Fields.By.label:
                self.key = self.param_type.Fields.By.key
                self.min_value = self.param_type.Fields.By.min_value
                self.value = self.param_type.Fields.By.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['By']
            elif self.label == BasePlate.Fields.Mx.label:
                self.key = self.param_type.Fields.Mx.key
                self.value = self.param_type.Fields.Mx.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['mx']
            elif self.label == BasePlate.Fields.My.label:
                self.key = self.param_type.Fields.My.key
                self.value = self.param_type.Fields.My.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['my']
            elif self.label == BasePlate.Fields.Mz.label:
                self.key = self.param_type.Fields.Mz.key
                self.value = self.param_type.Fields.Mz.value
                if self.key not in st.session_state:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index]['mz']
        self.render_widget()

    def render_widget(self):
        """Render the widget in Streamlit and handle synchronization with session state"""
        self._update_widgets_on_project_load()

        if not self.visible:
            st.write("None")
            return None
        if self.key:
            self._widget_sync_load(self.key)
        else:
            st.write("Bad Widget") # Debugging aid

    def _update_counter(self):
        st.session_state['global_widget_version_counter'] += 1
        return st.session_state['global_widget_version_counter']

    def _widget_sync_load(self, data_column_key: str):
        """Load widgets and keep them in sync with the data column"""
        dynamic_index: int = 0
        if self.widget_type:
            widget = self.WIDGET_TYPES[self.widget_type]

            # Map widget key to data column key
            data_key_mapping = {
                'fc': 'fc',
                'cracked_concrete': 'cracked_concrete',
                'grouted': 'grouted',
                'weight_classification_base': 'weight_classification_base',
                'poisson': 'poisson',
                't_slab': 't_slab',
                'profile': 'profile',
                'cx_neg': 'cx_neg',
                'cx_pos': 'cx_pos',
                'cy_neg': 'cy_neg',
                'cy_pos': 'cy_pos',
                'anchor_position': 'anchor_position',
                'deck_location': 'deck_location',
                'hole_diameter': 'hole_diameter',
                'face_side': 'face_side',
                'manufacturer': 'manufacturer',
                'specified_product': 'specified_product',
                'location': 'location',
                'seismic': 'seismic',
                'phi_override': 'phi_override',
                'Bx': 'Bx',
                'By': 'By',
                'mx': 'mx',
                'my': 'my',
                'mz': 'mz'
            }

            data_col_key = data_key_mapping[self.key]

            if self.key not in st.session_state:
                if data_col_key in st.session_state['data_column'][dynamic_index]:
                    st.session_state[self.key] = st.session_state['data_column'][dynamic_index][data_col_key]

            if st.session_state['global_widget_version_counter'] > st.session_state['global_version_counter']:
                if data_col_key in st.session_state['data_column'][dynamic_index]:
                    st.session_state['data_column'][dynamic_index][data_col_key] = st.session_state[self.key]
                st.session_state['global_version_counter'] = st.session_state['global_widget_version_counter']

            if st.session_state['global_widget_version_counter'] == st.session_state['global_version_counter']:
                # Build widget kwargs based on widget type
                widget_kwargs = {
                    'label': self.label,
                    'key': self.key,
                    'on_change': self._update_counter
                }

                if self.widget_type == 'selectbox':
                    widget_kwargs['options'] = self.options
                    if self.index is not None:
                        widget_kwargs['index'] = self.index
                    if self.placeholder:
                        widget_kwargs['placeholder'] = self.placeholder
                    if self.format_func:
                        widget_kwargs['format_func'] = self.format_func

                elif self.widget_type == 'number_input':
                    if self.min_value is not None:
                        widget_kwargs['min_value'] = self.min_value
                    if self.max_value is not None:
                        widget_kwargs['max_value'] = self.max_value
                    if self.value is not None:
                        widget_kwargs['value'] = self.value
                    if self.placeholder:
                        widget_kwargs['placeholder'] = self.placeholder

                widget(**widget_kwargs)

                if st.session_state[self.key] != st.session_state['data_column'][dynamic_index].get(data_col_key):
                    st.session_state['data_column'][dynamic_index][data_col_key] = st.session_state[self.key]

                    # Special handling for weight classification
                    if self.key == 'weight_classification_base':
                        st.session_state['data_column'][dynamic_index]['lw_factor'] = \
                            SubstrateParams.weight_class_lambda(st.session_state[self.key])

    def _update_widgets_on_project_load(self):
        """Update the widget value in session state with loaded values"""
        if st.session_state['global_version_counter'] > st.session_state['global_widget_version_counter']:
            for data_name in st.session_state['data_column'][0].index:
                if data_name in st.session_state:
                    st.session_state[data_name] = st.session_state['data_column'][0][data_name]
            st.session_state['global_widget_version_counter'] = st.session_state['global_version_counter']


def check_diff(design_index: int) -> bool:
    """Return True if any anchor key's current value differs from the design value.

    Raises:
        DataColumnError: if data_column/design_index is invalid or required keys are missing.
    """
    ss = st.session_state

    # Validate container and index
    try:
        data_col = ss['data_column']
    except KeyError as e:
        raise DataColumnError("Key 'data_column' not found in session state.") from e

    try:
        design = data_col[design_index]
    except IndexError as e:
        raise DataColumnError(f"design_index {design_index} out of range.") from e

    # Validate required keys exist in both places
    keys_to_check = [k for k in ANCHOR_PRO_BACKEND_PARAM_SS_KEYS if k != 'lw_factor']
    missing = [
        k for k in keys_to_check
        if k not in ss or k not in design
    ]
    if missing:
        raise DataColumnError(f"Missing keys in session or design: {missing}")

    # Return whether any key differs
    return any(ss[k] != design[k] for k in ANCHOR_PRO_BACKEND_PARAM_SS_KEYS)



def copy_design_params_to_widgets(design_index: int):
    """Update the data column with the current widget value"""
    if check_diff(design_index):
        for anchor_key in ANCHOR_PRO_BACKEND_PARAM_SS_KEYS:
            st.session_state[anchor_key] = st.session_state['data_column'][design_index][anchor_key]
    else:
        pass