"""Stores and manages data used with design editor"""
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, Optional, List, ClassVar
import streamlit as st
from utils.data_loader import load_anchor_spec_sheet, get_manufacturers
import pandas as pd
import streamlit as st
from pathlib import Path

@dataclass
class Params:
    """Base class for parameter groups"""
    pass


@dataclass
class SubstrateParams(Params):
    """Stores substrate parameters for user editing"""
    fc: int = 2000
    weight_classification_base: str = "NWC"
    poisson: float = 0.2
    t_slab: float = 12.0
    cx_neg: float = 5.0
    cx_pos: float = 5.0
    cy_neg: float = 5.0
    cy_pos: float = 5.0
    profile: str = "Slab"
    anchor_position: str = "top"
    grouted: str = "Grouted"
    deck_location: str = "Top"
    hole_diameter: float = 3.0
    face_side: str = "Face"
    cracked_concrete: str = "Cracked"
    lw_factor: float = 1.0

    # ---------- UI metadata as nested classes ----------
    class Fields:
        class BaseMaterial:
            label: ClassVar[str] = "Base Material"
            options: ClassVar[tuple[int, ...]] = (2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 8500)
            key: ClassVar[str] = "fc"

        class WeightClass:
            label: ClassVar[str] = "NWC / LWC"
            options: ClassVar[tuple[str, ...]] = ("NWC", "LWC")
            index: ClassVar[int] = 0
            key: ClassVar[str] = "weight_classification_base"

        class Poisson:
            label: ClassVar[str] = "poisson"
            min_value: ClassVar[float] = 0.0
            max_value: ClassVar[float] = 1.0
            value: ClassVar[float] = 0.2
            key: ClassVar[str] = "poisson"

        class ConcreteThickness:
            label: ClassVar[str] = "Concrete Thickness(in)"
            min_value: ClassVar[float] = 0.0
            value: ClassVar[float] = 12.0
            key: ClassVar[str] = "t_slab"

        class Profile:
            label: ClassVar[str] = "Slab / Filled Deck"
            options: ClassVar[tuple[str, ...]] = ("Slab", "Filled Deck")
            index: ClassVar[int] = 0
            key: ClassVar[str] = "profile"

        class EdgeDistXNeg:
            label: ClassVar[str] = "Edge dist -x"
            min_value: ClassVar[float] = 0.0
            value: ClassVar[float] = 5.0
            key: ClassVar[str] = "cx_neg"

        class EdgeDistXPos:
            label: ClassVar[str] = "Edge dist +x"
            min_value: ClassVar[float] = 0.0
            value: ClassVar[float] = 5.0
            key: ClassVar[str] = "cx_pos"

        class EdgeDistYNeg:
            label: ClassVar[str] = "Edge dist -y"
            min_value: ClassVar[float] = 0.0
            value: ClassVar[float] = 5.0
            key: ClassVar[str] = "cy_neg"

        class EdgeDistYPos:
            label: ClassVar[str] = "Edge dist +y"
            min_value: ClassVar[float] = 0.0
            value: ClassVar[float] = 5.0
            key: ClassVar[str] = "cy_pos"

        class Grouted:
            label: ClassVar[str] = "Grouted / Not-grouted"
            options: ClassVar[tuple[str, ...]] = ("Grouted", "Not-grouted")
            placeholder: ClassVar[str] = "Select..."
            key: ClassVar[str] = "grouted"

        class CrackedConcrete:
            label: ClassVar[str] = "Cracked / Uncracked"
            options: ClassVar[tuple[str, ...]] = ("Cracked", "Uncracked")
            index: ClassVar[int] = 0
            key: ClassVar[str] = "cracked_concrete"

        class AnchorPosition:
            label: ClassVar[str] = "Anchor Position"
            options: ClassVar[tuple[str, ...]] = ("top", "soffit")
            placeholder: ClassVar[str] = "Select..."
            key: ClassVar[str] = "anchor_position"

        class DeckLocation:
            label: ClassVar[str] = "Deck Installation Location"
            options: ClassVar[tuple[str, ...]] = ("Top", "Upper Flute", "Lower Flute")
            placeholder: ClassVar[str] = "Select..."
            key: ClassVar[str] = "deck_location"

        class HoleDiameter:
            label: ClassVar[str] = "Hole Diameter of Fastened Part"
            min_value: ClassVar[float] = 0.0
            placeholder: ClassVar[str] = "Input..."
            value: ClassVar[float] = 3.0
            key: ClassVar[str] = "hole_diameter"

        class FaceSide:
            label: ClassVar[str] = "Face, Side"
            options: ClassVar[tuple[str, ...]] = ("Face", "Side", "Top")
            placeholder: ClassVar[str] = "Select..."
            key: ClassVar[str] = "face_side"

    # ---------- helpers ----------
    @staticmethod
    def weight_class_lambda(weight_class: str) -> float:
        """Determine lambda given concrete weight classification"""
        return 0.75 if weight_class == "LWC" else 1.0

    @classmethod
    def iter_field_classes(cls):
        """Yield (name, nested_class) for all field metadata classes."""
        for name, obj in vars(cls.Fields).items():
            if isinstance(obj, type) and obj.__module__ == cls.__module__:
                # basic filter to skip dunders and non-field attrs
                if hasattr(obj, "key") and hasattr(obj, "label"):
                    yield name, obj


@dataclass
class AnchorProduct(Params):
    """Stores anchor products for user editing"""
    mode: Optional[str] = None
    product_group: Optional[str] = None
    anchor_parameters: pd.DataFrame = field(default_factory=load_anchor_spec_sheet)
    specified_product: Optional[str] = field(
        default_factory=lambda: load_anchor_spec_sheet().iloc[0]["anchor_id"]
    )

    # ---------- UI metadata as nested classes ----------
    class Fields:
        class Manufacturer:
            label: ClassVar[str] = "Manufacturer"
            # `options` is dynamic, filled after init
            options: tuple = ()
            placeholder: ClassVar[str] = "Select..."
            key: ClassVar[str] = "manufacturer"

        class SpecifiedProduct:
            label: ClassVar[str] = "Specified Product"
            placeholder: ClassVar[str] = "Select..."
            key: ClassVar[str] = "specified_product"
            index: ClassVar[int] = 1

    # ---------- lifecycle ----------
    def __post_init__(self):
        # Populate manufacturer options dynamically
        self.anchor_manufacturer: tuple = get_manufacturers(self.anchor_parameters)
        self.Fields.Manufacturer.options = self.anchor_manufacturer

    # ---------- helpers ----------
    @classmethod
    def iter_field_classes(cls):
        """Yield (name, nested_class) for all field metadata classes."""
        for name, obj in vars(cls.Fields).items():
            if isinstance(obj, type) and obj.__module__ == cls.__module__:
                if hasattr(obj, "key") and hasattr(obj, "label"):
                    yield name, obj


@dataclass
class LoadingParams(Params):
    """Stores loading parameters for user editing"""
    location: str = "Individual Anchors"
    seismic: bool = True
    phi_override: bool = False

    # ---------- UI metadata as nested classes ----------
    class Fields:
        class LoadLocation:
            label: ClassVar[str] = "Anchor Load Input Location"
            options: ClassVar[tuple[str, str]] = ("Individual Anchors", "Group Origin")
            placeholder: ClassVar[str] = "Select..."
            index: ClassVar[int] = 0
            key: ClassVar[str] = "location"

        class Seismic:
            label: ClassVar[str] = "Seismic Loading"
            options: ClassVar[tuple[bool, bool]] = (True, False)
            placeholder: ClassVar[str] = "Select..."
            index: ClassVar[int] = 0
            key: ClassVar[str] = "seismic"

        class PhiOverride:
            label: ClassVar[str] = "Phi Override"
            placeholder: ClassVar[str] = "Select..."
            options: ClassVar[tuple[bool, bool]] = (True, False)
            index: ClassVar[int] = 1
            key: ClassVar[str] = "phi_override"

            @staticmethod
            def format_options(option: bool) -> str:
                """Format boolean options for display"""
                return "Yes" if option else "No"

    # ---------- helpers ----------
    @classmethod
    def iter_field_classes(cls):
        """Yield (name, nested_class) for all field metadata classes."""
        for name, obj in vars(cls.Fields).items():
            if isinstance(obj, type) and hasattr(obj, "key") and hasattr(obj, "label"):
                yield name, obj


@dataclass
class InstallationParams(Params):
    """Stores installation parameters for user editing"""
    hef: Optional[float] = None
    short_term_temp: Optional[float] = None
    long_term_temp: Optional[float] = None
    drilling_type: Optional[str] = None
    inspection_condition: Optional[str] = None
    moisture_condition: Optional[str] = None


@dataclass
class Anchor(Params):
    """Stores anchor geometry data"""
    # Lists to store anchor geometry points
    anchor_geometry_forces: Optional[pd.DataFrame] = field(default_factory=lambda: pd.DataFrame({
        'X': [0.0, 0.0, -6.0, -6.0],
        'Y': [0.0, 6.0, 6.0, 0.0],
        'Vx': [0.0, 0.0, 0.0, 0.0],
        'Vy': [0.0, 0.0, 0.0, 0.0],
        'N': [500, 500, 500, 500]}
        ))
    anchor_count: Optional[int] = 2


@dataclass
class BasePlate(Params):
    Bx: float = 24.0
    By: float = 24.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0

    class Fields:
        class Bx:
            label: ClassVar[str] = "Base Plate Width (in)"
            min_value: ClassVar[float] = 0.0
            value: ClassVar[float] = 24.0
            key: ClassVar[str] = "Bx"

        class By:
            label: ClassVar[str] = "Base Plate Length (in)"
            min_value: ClassVar[float] = 0.0
            value: ClassVar[float] = 24.0
            key: ClassVar[str] = "By"

        class Mx:
            label: ClassVar[str] = "Moment X (lb-in)"
            value: ClassVar[float] = 0.0
            key: ClassVar[str] = "mx"

        class My:
            label: ClassVar[str] = "Moment Y (lb-in)"
            value: ClassVar[float] = 0.0
            key: ClassVar[str] = "my"

        class Mz:
            label: ClassVar[str] = "Moment Z (lb-in)"
            value: ClassVar[float] = 0.0
            key: ClassVar[str] = "mz"


@dataclass
class DesignParameters(Params):
    """Stores design editor data used in calculations"""
    substrate: SubstrateParams = field(default_factory = SubstrateParams)
    anchor_product: AnchorProduct = field(default_factory = AnchorProduct)
    loading: LoadingParams = field(default_factory = LoadingParams)
    installation: InstallationParams = field(default_factory = InstallationParams)
    anchor: Anchor = field(default_factory = Anchor)
    baseplate: BasePlate = field(default_factory = BasePlate)

    parameters: list = field(default_factory = list)
    combined_dict: Optional[dict] = None


    def collect_parameter_names(self) -> None:
        """Collects all attributes that will be used input to anchor pro"""
        self.parameters = (
            [key.name for key in fields(self.substrate)] +
            [key.name for key in fields(self.anchor_product)] +
            [key.name for key in fields(self.loading)] +
            [key.name for key in fields(self.installation)] +
            [key.name for key in fields(self.anchor)] +
            [key.name for key in fields(self.baseplate)])

    def parameters_to_dict(self) -> None:
        """Converts editor attributes to dict"""

        substrate_dict = asdict(self.substrate)
        anchor_product_dict = asdict(self.anchor_product)
        loading_dict = asdict(self.loading)
        installation_dict = asdict(self.installation)
        anchor_geometry_dict = asdict(self.anchor)
        baseplate_dict = asdict(self.baseplate)

        self.combined_dict = (substrate_dict | anchor_product_dict | loading_dict | 
                              installation_dict | anchor_geometry_dict | baseplate_dict)

    def __post_init__(self):
        self.collect_parameter_names()
        self.parameters_to_dict()
