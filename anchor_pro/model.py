from dataclasses import dataclass, field, fields
from typing import Optional, Union, List, Tuple
from numpy.typing import NDArray
from enum import Enum

import numpy as np
import math

import warnings
import os
import json
import pandas as pd
from numpy.lib.mixins import NDArrayOperatorsMixin
from scipy.optimize import root, broyden1, newton_krylov


import anchor_pro.elements.base_plates as bp
# import anchor_pro.elements.base_straps  #todo [BASE STRAPS]
import anchor_pro.elements.wall_bracket as wbkt
import anchor_pro.elements.wall_backing as wbing
import anchor_pro.elements.concrete_anchors as conc
import anchor_pro.elements.wood_fasteners as wf
import anchor_pro.elements.sms as sms
import anchor_pro.elements.fastener_connection as cxn
from anchor_pro.ap_types import (FactorMethod, WallPositions)

import anchor_pro.elements.base_plates

'''ENUMERATION DEFINITIONS'''
class Codes(str, Enum):
    asce7_16 = "ASCE 7-16"
    asce7_22 = "ASCE 7-22"
    asce7_22_OPM = "ASCE 7-22 OPM"
    cbc98 = "CBC 1998, 16B"

class InstallType(str, Enum):
    base = "Base Anchored"
    brace = "Wall Brackets"
    wall = "Wall Mounted"
    suspended = "Suspended"

class BaseMaterial(str, Enum):
    concrete = "Concrete"
    wood = "Wood"

class WallMaterial(str, Enum):
    concrete = "Concrete"
    cmu = "CMU"
    wood = "Wood Stud"
    cfs = "MetalStud"

@dataclass(frozen=True,slots=True)
class EquipmentInfo:
    equipment_id: str

    equipment_type: str

@dataclass(frozen=True, slots=True)
class ASCE7_16Pars:
    ap: float
    Rp: float
    Ip: float
    sds: float
    z: float
    h: float
    omega: float
    use_dynamic: bool = False  # Future feature toggle
    ai: Optional[float] = None
    Ax: Optional[float] = None
    code_edition: Codes = Codes.asce7_16

@dataclass(frozen=True, slots=True)
class ASCE7_22_OPMPars:
    Cpm: float
    Cv: float
    omega: float
    code_edition: Codes = Codes.asce7_22_OPM

@dataclass(frozen=True, slots=True)
class CBC_98Pars:
    Ip: float
    Z: float
    Cp: float
    cp_amplification: int
    cp_category: str
    below_grade: bool
    grade_factor: float
    Cp_eff: Optional[float] = None  # set post-init or remain None
    code_edition: Codes = Codes.cbc98

CODE_PARS = Union[ASCE7_16Pars, ASCE7_22_OPMPars, CBC_98Pars]

@dataclass(frozen=True, slots=True)
class FpCalc_asce7_16:
    Fp: float
    Fp_min: float
    Fp_max: float
    Fp_code: float
    Eh: float
    Emh: float
    Ev: float
    use_dynamic: bool = False
    Fp_dynamic: Optional[float] = None

@dataclass(frozen=True, slots=True)
class FpCalc_asce7_22:
    """
    Placeholder for a future ASCE 7-22 calc (non-OPM).
    Fields mirror ASCE 7-16 for now; adjust when the 7-22 formulation is implemented.
    """
    Fp: float
    Fp_min: float
    Fp_max: float
    Fp_code: float
    Eh: float
    Emh: float
    Ev: float

@dataclass(frozen=True, slots=True)
class FpCalc_asce7_22_OPM:
    Fp: float
    Eh: float
    Emh: float
    Ev: float

@dataclass(frozen=True, slots=True)
class FpCalc_cbc98:
    Fp: float
    Eh: float
    Emh: float
    Ev: float
    Cp_eff: float

# Optional convenience union for the return type
FP_CALC = Union[FpCalc_asce7_16, FpCalc_asce7_22, FpCalc_asce7_22_OPM, FpCalc_cbc98]

@dataclass(frozen=True, slots=True)
class FactoredLoadCase:
    """A single load case (e.g., ASD, LRFD, or LRFD with Ω-level)."""
    name: FactorMethod
    Fh: float         # Factored horizontal force
    Fv_min: float     # Most-positive (less downward)
    Fv_max: float     # Most-negative (more downward)
    Fv: float         # Selected vertical force based on instal type

@dataclass(frozen=True, slots=True)
class FactoredLoads:
    """Bundle of all load cases for a given Fp result."""
    asd: Optional[FactoredLoadCase]
    lrfd: Optional[FactoredLoadCase]
    lrfd_omega: Optional[FactoredLoadCase]

    def get(self, method: FactorMethod, use_fv_max: bool=True) -> Tuple[float, float]:
        """Return (Fh, Fv) for a requested factor type and support condition."""
        mapping = {FactorMethod.asd: self.asd, FactorMethod.lrfd: self.lrfd, FactorMethod.lrfd_omega: self.lrfd_omega}
        case = mapping[method]
        if case is None:
            raise ValueError(f"Load case {method} is not available for this code edition.")
        return case.Fh, case.Fv

@dataclass(frozen=True,slots=True)
class EquipmentProps:
    Wp: float
    Bx: float
    By: float
    H: float
    zCG: float
    ex: float
    ey: float

    gauge: Optional[float] = None
    fy: Optional[float] = None

@dataclass(frozen=True,slots=True)
class WallOffsets:
    XP: float
    XN: float
    YP: float
    YN: float

@dataclass(frozen=True,slots=True)
class InstallationInfo:
    installation_type: InstallType
    base_material: Optional[BaseMaterial] = None
    wall_material: Optional[WallMaterial] = None #formerly called wall_type
    wall_offsets: Optional[WallOffsets] = None


BASE_ANCHOR_ELEMENTS = Union[
        anchor_pro.elements.concrete_anchors.ConcreteAnchors,
        anchor_pro.elements.wood_fasteners.WoodFastener]

BASE_ANCHOR_RESULTS = Union[
        anchor_pro.elements.concrete_anchors.ConcreteAnchorResults,
        anchor_pro.elements.wood_fasteners.WoodFastenerResults]

WALL_ANCHOR_ELEMENTS = Union[
    anchor_pro.elements.concrete_anchors.ConcreteAnchors,
    anchor_pro.elements.wood_fasteners.WoodFastener,
    anchor_pro.elements.sms.SMSAnchors]

WALL_ANCHOR_RESULTS = Union[
    anchor_pro.elements.concrete_anchors.ConcreteAnchorResults,
    anchor_pro.elements.wood_fasteners.WoodFastenerResults,
    anchor_pro.elements.sms.SMSResults]

@dataclass(frozen=True,slots=True)
class Elements:
    """ All entries should be one of two types:
    1) A list of element objects
    2) A list of indices constituting a mapping between connected elements.
    Must provide metadata tag is_element."""
    # Base Anchor Elements
    base_plates: List[anchor_pro.elements.base_plates.BasePlateElement] = field(default_factory=list, metadata={"is_element":True})
    base_anchors: List[BASE_ANCHOR_ELEMENTS] = field(default_factory=list,metadata={"is_element":True})
    base_plate_connections: List[anchor_pro.elements.fastener_connection.FastenerConnection] = field(default_factory=list,metadata={"is_element":True})
    base_plate_fasteners: List[anchor_pro.elements.sms.SMSAnchors] = field(default_factory=list,metadata={"is_element":True})
    # base_straps: List[anchor_pro.elements.base_straps.BaseStrapElement] = field(default_factory=list) #todo[base straps]

    # Base Anchor Maps
    bp_to_cxn: List[int] = field(default_factory=list,metadata={"is_element":False})  # Base Plate to Connections Map
    cxn_to_sms: List[int] = field(default_factory=list, metadata={"is_element":False})  # Connection to SMS Map
    sms_to_cxn: List[int] = field(default_factory=list, metadata={"is_element":False})
    bp_to_anchors: List[int] = field(default_factory=list, metadata={"is_element":False})  # Base Plate to Base Anchor Map
    anchors_to_bp: List[List[int]] = field(default_factory=list, metadata={"is_element":False})

    # Wall Anchor Elements
    wall_brackets: List[anchor_pro.elements.wall_bracket.WallBracketElement] = field(default_factory=list,metadata={"is_element":True})
    wall_backing: List[anchor_pro.elements.wall_backing.WallBackingElement] = field(default_factory=list,metadata={"is_element":True})
    wall_anchors: List[WALL_ANCHOR_ELEMENTS] = field(default_factory=list,metadata={"is_element":True})
    wall_bracket_connections: List[anchor_pro.elements.fastener_connection.FastenerConnection] = field(
        default_factory=list, metadata={"is_element": True})
    wall_bracket_fasteners: List[anchor_pro.elements.sms.SMSAnchors] = field(
        default_factory=list, metadata={"is_element": True})

    # Wall Anchor Maps
    bracket_to_cxn: List[int] = field(default_factory=list, metadata={"is_element":False})
    cxn_to_bracket: List[int] = field(default_factory=list, metadata={"is_element":False})
    wcxn_to_sms: List[int] = field(default_factory=list, metadata={"is_element":False})
    sms_to_wcxn: List[int] = field(default_factory=list, metadata={"is_element":False})
    bracket_to_backing: List[int] = field(default_factory=list, metadata={"is_element":False})
    backing_to_brackets: List[List[int]] = field(default_factory=list, metadata={"is_element":False})
    backing_to_anchors: List[int] = field(default_factory=list, metadata={"is_element":False})
    anchors_to_backing: List[List[int]] = field(default_factory=list, metadata={"is_element":False})



    def __post_init__(self):
        # Anchor to Base Plate (not one to one)
        anchors_to_bp = []
        for a_idx, anchors in enumerate(self.base_anchors):
            anchors_to_bp.append([bp_idx for bp_idx, anch_idx in enumerate(self.bp_to_anchors) if anch_idx==a_idx])
        object.__setattr__(self,"anchors_to_bp",anchors_to_bp)

        # SMS Object to Connection (One to one)
        sms_to_cxn = [cxn_idx for cxn_idx, sms_idx in enumerate(self.cxn_to_sms)]
        object.__setattr__(self, "sms_to_cxn", sms_to_cxn)



@dataclass(slots=True)
class AnalysisVars:
    factor_methods : set
    omit_analysis: bool = False
    model_unstable: bool = False
    cor_coordinates: Optional[dict] = None
    u_previous: Optional[NDArray] = None
    K: Optional[NDArray] = None
    residual_call_count: int = 0

@dataclass(frozen=True, slots=True)
class DofProps:
    n_dof: int = 6
    disp_dofs: List[int] = field(default_factory=list)
    base_plate_dofs: Optional[NDArray] = None # DOF connectivity array for floor plate elements

@dataclass(frozen=True, slots=True)
class Solution:
    theta_z: NDArray
    equilibrium_solutions: NDArray
    converged: List[bool]
    factor_method: FactorMethod

@dataclass(frozen=True, slots=True)
class ElementResults:
    base_plates: Optional[List[bp.BasePlateResults]] = field(default_factory=list)
    base_anchors: Optional[List[BASE_ANCHOR_RESULTS]] = field(default_factory=list)
    base_plate_connections: Optional[List[cxn.ElasticBoltGroupResults]] = field(default_factory=list)
    base_plate_fasteners: Optional[List[sms.SMSResults]] = field(default_factory=list)
    wall_brackets: Optional[List[wbkt.WallBracketResults]] = field(default_factory=list)
    wall_backing: Optional[List[cxn.ElasticBoltGroupResults]] = field(default_factory=list)
    wall_anchors: Optional[List[WALL_ANCHOR_RESULTS]] = field(default_factory=list)
    wall_bracket_connections: Optional[List[cxn.ElasticBoltGroupResults]] = field(default_factory=list)
    wall_bracket_fasteners: Optional[List[cxn.ElasticBoltGroupResults]] = field(default_factory=list)



    # Governing indices/unities
    @property
    def max_unity(self):
        unities_across_elements = []
        for f in fields(self):
            el_result = getattr(self, f.name, None)
            if el_result:
                unity = max([getattr(element, 'unity', -np.inf) for element in el_result])
                unities_across_elements.append(unity)
        return max(unities_across_elements)

def get_needed_factor_methods(elements: Elements) -> set:
    """
    Returns (need_lrfd, need_asd, need_lrfd_omega) by scanning only fields
    whose metadata marks them as element lists: metadata["is_element"] == True.
    """
    needed_methods: set[FactorMethod] = set()

    for f in fields(elements):
        if not f.metadata.get("is_element", False):
            continue

        lst = getattr(elements, f.name)
        if not lst:  # empty list, skip quickly
            continue

        for el in lst:
            needed_methods.add(el.factor_method)

        # Early exit if all satisfied
        if len(needed_methods) == len(FactorMethod.__members__):
            break

    return needed_methods

def make_code_pars(code: Codes, project_info: dict, equipment_data: dict) -> CODE_PARS:
    if code == Codes.cbc98:
        return CBC_98Pars(
            Ip=project_info['Ip'],
            Z=project_info['Z'],
            Cp=equipment_data['Cp'],
            cp_amplification=equipment_data['cp_amplification'],
            cp_category=equipment_data['cp_category'],
            below_grade=equipment_data['below_grade'],
            grade_factor=1 if not equipment_data['below_grade'] else 2/3,
            Cp_eff=None
        )
    elif code == Codes.asce7_16:
        return ASCE7_16Pars(
            ap=equipment_data['ap'],
            Rp=equipment_data['Rp'],
            Ip=equipment_data['Ip'],
            sds=project_info['sds'],
            z=equipment_data['z'],
            h=equipment_data['building_height'],
            omega=equipment_data['omega'],
            use_dynamic=False,
            ai=None,
            Ax=None
        )
    elif code == Codes.asce7_22:
        return ASCE7_22_OPMPars(
            Cpm=equipment_data['Cpm'],
            Cv=equipment_data['Cv'],
            omega=equipment_data['omega_opm']
        )
    else:
        raise ValueError(f"Unknown code edition: {code}")

class EquipmentModel:
    def __init__(
            self,
            equipment_info: EquipmentInfo,
            install: InstallationInfo,
            code_pars: CODE_PARS,
            equipment_props: EquipmentProps,
            elements: Elements,
            wall_offsets: WallOffsets
            ):
        self.equipment_info = equipment_info
        self.code_pars = code_pars
        self.equipment_props = equipment_props
        self.install = install
        self.elements = elements
        self.dof_props = self.number_degrees_of_freedom()
        self.fp_calc = self.calculate_fp()
        self.factored_loads = self.calculate_factored_loads(self.fp_calc)
        self.analysis_vars = AnalysisVars(factor_methods= get_needed_factor_methods(self.elements))
        self.wall_offsets = wall_offsets

        #Unresolved Parameters
        # todo: include_overstrength (component-level?), factor_type (component-level?) or both. Have model read it's componants and know which load cases it needs to run
        self.cxn_sms_id = None # move to hardware selections
        self.include_overstrength = None  # move to elements
        self.factor_type = None  # move to elements
        #base_anchor_id # move to hardware selections
        #base_anchor_groups # move to hardware selections

        self._num_base_anchors_in_tension = 0  # Used for stiffness update logic in solver

    def number_degrees_of_freedom(self):
        """Numbers the degrees of freedom, including additional DOFs for floor plates with moment releases."""
        base_plate_dofs = np.full((len(self.elements.base_plates), 6), [0, 1, 2, 3, 4, 5], dtype=int)
        dof_count = 6
        disp_dofs = [1, 1, 1, 0, 0, 0]
        for i, element in enumerate(self.elements.base_plates):
            props = element.props
            rel = props.releases
            if rel.zp and rel.zn:
                base_plate_dofs[i, 2] = dof_count
                disp_dofs.append(1)
                dof_count += 1
            if rel.mx: # or element.prying_mx:
                base_plate_dofs[i, 3] = dof_count
                disp_dofs.append(0)
                dof_count += 1
            if rel.my: # or element.prying_my:
                base_plate_dofs[i, 4] = dof_count
                disp_dofs.append(0)
                dof_count += 1
            if rel.mz:
                base_plate_dofs[i, 5] = dof_count
                disp_dofs.append(0)
                dof_count += 1
        n_dof = dof_count

        # Set Element Dofs
        for i, element in enumerate(self.elements.base_plates):
            element.dof_props = bp.create_dof_constraints(
                n_dof=n_dof,
                dof_map = base_plate_dofs[i, :],
                props=element.props)
        # todo [Wall Brackets]
        # for element in self.wall_brackets:
        #     element.set_dof_constraints(self.n_dof)
        # todo [Straps]
        # for strap in self.base_straps:
        #     strap.pre_compute_matrices()


        return DofProps(
            n_dof = n_dof,
            disp_dofs = disp_dofs,
            base_plate_dofs=base_plate_dofs)

    def set_model_data(
        self,
        base_anchor_props:List[conc.MechanicalAnchorProps],
        base_plate_stiffness: List[bp.AnchorStiffness],
        wall_anchor_props: List[conc.MechanicalAnchorProps],
        wall_bracket_props: List[wbkt.BracketProps],
        cxn_sms_size, sms_catalog):

        els = self.elements

        # Base Anchors and Base Plate Stiffness
        for el, props in zip(els.base_anchors, base_anchor_props):
            el.set_anchor_props(props)
            #todo: el.check_anchor_spacing()
            # if not all(self.elements.base_anchors.spacing_requirements.values()):
            #     self._analysis_vars.omit_analysis = True
        if els.base_anchors:
            for el, prop_idx in zip(els.base_plates, els.bp_to_anchors):
                el.set_anchor_stiffness(base_plate_stiffness[prop_idx])

        # Base Plate Fasteners
        for el in els.base_plate_fasteners:
            if isinstance(el, sms.SMSAnchors):
                el.set_screw_size(sms_catalog, cxn_sms_size)

        # Wall Anchors
        for el, props in zip(els.wall_anchors, wall_anchor_props):
            el.set_anchor_props(props)
            #todo: el.check_anchor_spacing()
            # if not all(self.elements.base_anchors.spacing_requirements.values()):
            #     self._analysis_vars.omit_analysis = True

        # Wall Brackets
        for el, props in zip(els.wall_brackets, wall_bracket_props):
            el.set_bracket_props(props)

        # Bracket Connections
        for el in els.wall_bracket_fasteners:
            if isinstance(el, sms.SMSAnchors):
                el.set_screw_size(sms_catalog, cxn_sms_size)


    def calculate_fp(self, use_dynamic: bool = False) -> FP_CALC:
        # ASCE 7-16
        if self.code_pars.code_edition == Codes.asce7_16:
            sds = self.code_pars.sds
            Ip = self.code_pars.Ip
            ap = self.code_pars.ap
            Rp = self.code_pars.Rp
            z = self.code_pars.z
            h = self.code_pars.h
            omega = self.code_pars.omega
            Wp = self.equipment_props.Wp

            Fp_min = 0.3 * sds * Ip * Wp  # ASCE7-16 13.3-2
            Fp_max = 1.6 * sds * Ip * Wp  # ASCE7-16 13.3-3
            Fp_code = (0.4 * ap * sds * Wp * Ip / Rp) * (1 + 2 * z / h)  # 13.3-1

            if use_dynamic:
                ai = self.code_pars.ai
                Ax = self.code_pars.Ax
                Fp_dynamic = (ap * Wp * Ip / Rp) * ai * Ax  # 13.3-4
                Fp = max(min(Fp_max, Fp_dynamic), Fp_min)  # 13.3.1.1
            else:
                Fp_dynamic = None
                Fp = max(min(Fp_max, Fp_code), Fp_min)  # 13.3.1.1

            Eh = Fp  # 13.3.1.1
            Emh = omega * Eh
            Ev = 0.2 * sds * Wp  # 13.3.1.2

            return FpCalc_asce7_16(
                Fp=Fp, Fp_min=Fp_min, Fp_max=Fp_max, Fp_code=Fp_code,
                Eh=Eh, Emh=Emh, Ev=Ev, use_dynamic=use_dynamic, Fp_dynamic=Fp_dynamic
            )

        # ASCE 7-22 (non-OPM) not yet implemented
        elif self.code_pars.code_edition == Codes.asce7_22:
            raise NotImplementedError('Definition of Fp for ASCE 7-22 (non-OPM) not yet defined')

        # ASCE 7-22 OPM
        elif self.code_pars.code_edition == Codes.asce7_22_OPM:
            Wp = self.equipment_props.Wp
            Cpm = self.code_pars.Cpm
            Cv = self.code_pars.Cv
            omega = self.code_pars.omega

            Fp = Cpm * Wp
            Eh = Fp
            Emh = omega * Eh
            Ev = Cv * Wp

            return FpCalc_asce7_22_OPM(Fp=Fp, Eh=Eh, Emh=Emh, Ev=Ev)

        # CBC 1998
        elif self.code_pars.code_edition == Codes.cbc98:
            Z = self.code_pars.Z
            Ip = self.code_pars.Ip
            Cp = self.code_pars.Cp
            amp = self.code_pars.cp_amplification
            grade_factor = self.code_pars.grade_factor
            Wp = self.equipment_props.Wp

            max_amplification = {1: 999, 2: 2, 4: 3}
            Cp_eff = min(Cp * amp, max_amplification[amp]) * grade_factor

            Fp = Z * Ip * Cp_eff * Wp
            Eh = Fp
            Ev = Eh / 3
            Emh = Fp

            return FpCalc_cbc98(Fp=Fp, Eh=Eh, Emh=Emh, Ev=Ev, Cp_eff=Cp_eff)

        else:
            raise NotImplementedError(f'Specified code edition {self.code_pars.code_edition} not supported.')

    def calculate_factored_loads(self, fp) -> FactoredLoads:
        """
        Compute ASD, LRFD, and LRFD_OMEGA load cases for the current model.
        Inputs:
            self.Wp
            self.code_pars.code_edition
            fp: one of fp_calc_asce7_16 | fp_calc_asce7_22_OPM | fp_calc_cbc98 (with Eh, Ev, Emh)
        Returns:
            FactoredLoads(dataclass)
        """
        Wp = self.equipment_props.Wp
        ed = self.code_pars.code_edition

        # ASCE 7-16 and ASCE 7-22 OPM share the same factoring shown in your code
        if ed in (Codes.asce7_16, Codes.asce7_22_OPM):
            # LRFD vertical
            Fuv_min = -0.9 * Wp + 1.0 * fp.Ev
            Fuv_max = -1.2 * Wp - 1.0 * fp.Ev
            if self.install in [InstallType.base, InstallType.brace]:
                Fuv = Fuv_min
            else:
                Fuv = Fuv_max

            # LRFD horizontal (base: Eh), plus an explicit Omega variant (Emh)
            lrfd = FactoredLoadCase(FactorMethod.lrfd, Fh=1.0 * fp.Eh, Fv_min=Fuv_min, Fv_max=Fuv_max, Fv=Fuv)
            lrfd_omega = FactoredLoadCase(FactorMethod.lrfd_omega, Fh=1.0 * fp.Emh, Fv_min=Fuv_min, Fv_max=Fuv_max, Fv=Fuv)

            # ASD vertical
            Fav_min = -0.6 * Wp + 0.7 * fp.Ev
            Fav_max = -1.0 * Wp - 0.7 * fp.Ev
            if self.install.installation_type in [InstallType.base, InstallType.brace]:
                Fav = Fav_min
            else:
                Fav = Fav_max
            # ASD horizontal (base: Eh)
            asd = FactoredLoadCase(FactorMethod.asd, Fh=0.7 * fp.Eh, Fv_min=Fav_min, Fv_max=Fav_max, Fv = Fav)

            return FactoredLoads(asd=asd, lrfd=lrfd, lrfd_omega=lrfd_omega)

        elif ed == Codes.asce7_22:
            # Not defined in your current logic
            raise NotImplementedError('Definition of factored loads for ASCE 7-22 (non-OPM) not yet defined.')

        elif ed == Codes.cbc98:
            # CBC 1998, 16B
            # LRFD Combinations
            Fuv_min = -0.9 * Wp + 1.3 * 1.1 * fp.Ev # 9B-3
            Fuv_max = -0.75 * (1.4 * Wp + 1.7 * 1.1 * fp.Ev) # 9B-2
            if self.install.installation_type in [InstallType.base, InstallType.brace]:
                Fuv = Fuv_min
            else:
                Fuv = Fuv_max
            # LRFD horizontal
            lrfd = FactoredLoadCase(FactorMethod.lrfd, Fh=0.75 * 1.7 * 1.1 * fp.Eh,
                                    Fv_min=Fuv_min, Fv_max=Fuv_max, Fv=Fuv)
            # CBC 98 doesn't natively define an "omega-level" pathway; for symmetry, mirror LRFD:
            lrfd_omega = FactoredLoadCase(FactorMethod.lrfd_omega, Fh=0.75 * 1.7 * 1.1 * fp.Eh,
                                          Fv_min=Fuv_min, Fv_max=Fuv_max, Fv=Fuv)
            # ASD horizontal per your code (Fav vertical was commented in original — mirror LRFD verticals for completeness)
            Fav_min = -0.9*Wp + 1.0*fp.Ev
            Fav_max = -1.0*Wp +1.0*fp.Ev

            asd = FactoredLoadCase(FactorMethod.asd, Fh=fp.Eh, Fv_min=Fuv_min, Fv_max=Fuv_max, Fv=Fuv)  #todo: QA vertical loads at ASD level for 98 CBC

            return FactoredLoads(asd=asd, lrfd=lrfd, lrfd_omega=lrfd_omega)

        else:
            raise Exception('Specified code year not supported.')

    def check_model_stability(self, tol=1e-12):
        """ Can be run after setting all anchor and hardware data.
        Will impose small dof displacements in principal directions
        and verify model stabilit by checking for non-zero eigenvalues."""

        dof_labels = ['Dx', 'Dy', 'Dz', 'Rx', 'Ry', 'Rz']
        dir_labels = ['+', '-']

        u0 = 1 * np.eye(self.dof_props.n_dof)
        for dof, u_dof in enumerate(u0[0:5, :]):
            for direction, u in enumerate([u_dof, -1*u_dof]):
                k = self.update_stiffness_matrix(u)
                # eigenvalues = np.linalg.eigenvalues(k)
                # zero_modes = np.sum(np.abs(eigenvalues) < tol)
                # unstable = zero_modes > 1
                p = k @ u
                unstable = np.abs(p[dof]) < tol

                if unstable:
                    if self.install.installation_type == InstallType.brace and dof == 2 and direction == 0:
                        '''Ignore Dz+ instability for Wall Brackets units.
                        It is assumed that resultant vertical forces will always be downward.'''
                        continue
                    else:
                        self.analysis_vars.model_unstable = self.analysis_vars.omit_analysis = True
                        warnings.warn(f'WARNING: Model instability detected at DOF {dof_labels[dof] + dir_labels[direction]}. '
                              f'Check model geometry definitions, material properties, and releases.')
                        return

        self.analysis_vars.cor_coordinates = self.get_center_of_rigidity()

    def get_center_of_rigidity(self):

        cor_coordinates = {'x_displacement': {1: None,
                                              -1: None},
                           'y_displacement': {1: None,
                                              -1: None},
                           'z_displacement': {1: None,
                                              -1: None}}
        axes = ['x_displacement', 'y_displacement', 'z_displacement']
        u_vecs = np.eye(self.dof_props.n_dof, 3)

        ''' Center of Rigidity coordinates are computed generally as m/p,
            where m is a relevant component of moment (mx, my, or mz).
            for a force in x-direction: cor = (0, -mz/px, my/px),
            for a force in y-direction: cor = (mz/py, 0, -mx / py),
            for a force in z-direction: cor = (-my/pz, mx / pz, 0).

            For conciseness in the code, this is computed by extracting the signs of the various terms,
            and the index of the relevant moment component into lists which can be iterated over.'''

        moment_term_signs = [np.array([0, -1, 1]),
                             np.array([1, 0, -1]),
                             np.array([-1, 1, 0])]
        moment_component_idx = [(0, 2, 1),
                                (2, 1, 0),
                                (1, 0, 2)]

        for i, axis in enumerate(axes):
            for dir_sign in [-1, 1]:
                u = dir_sign * u_vecs[:, i]
                k = self.update_stiffness_matrix(u)
                p = k @ u
                force = p[i]
                moments = p[3:]
                cor_coordinates[axis][dir_sign] = [sgn * moments[idx] / force for sgn, idx in
                                                   zip(moment_term_signs[i], moment_component_idx[i])]

        return cor_coordinates

    def update_stiffness_matrix(self, u):
        n_dof = self.dof_props.n_dof
        elems = self.elements
        k = np.zeros((n_dof, n_dof))
        # Base Plates
        #TODO: Huristically, it was determined that these stiffness terms needed to be negated.
        # Dig into why the signs for base plate stiffess terms needed to be reversed
        k += -sum(element.get_element_stiffness_matrix(u) for element in elems.base_plates)
        #todo: base straps
        # k += sum(element.get_element_stiffness_matrix(u) for element in self.base_straps)

        # Wall Brackets
        k_brackets = [element.get_element_stiffness_matrix(u) for element in elems.wall_brackets]
        if k_brackets:
            k[:6, :6] += np.sum(k_brackets, axis=0)

        return k



    def analyze(
            self,
            Fh: float,
            Fv: float,
            factor_method: FactorMethod,
            initial_solution_cache: Optional[Solution]=None,
            verbose: bool =False) -> Solution:

        """Applies Horizontal Loads at all angles, solves for equilibrium and stores the solution displacements"""
        n_dof = self.dof_props.n_dof

        # 1) Angle grid
        num_theta_z = 4 * 8 + 1
        theta_z = np.linspace(0, 2 * math.pi, num_theta_z)


        # 2) Load vectors for all theta
        loads = self.get_load_vector(Fh, Fv, theta_z)

        # 3) Storage
        converged: list[bool] = []
        equilibrium_solutions = np.zeros((n_dof, len(theta_z)), dtype=float)

        # 4) Initial-Guess Controls
        try_previous_converged = False
        u_prev = np.zeros(n_dof, dtype=float)

        # 5) Analysis Attempt with Initial DOF Guesses
        for i in range(theta_z.size):
            p = loads[:,i]
            t = theta_z[i]

            u_tries: list[np.ndarray] = []
            if (initial_solution_cache is not None) and initial_solution_cache.converged[i]:
                u_tries.append(initial_solution_cache.equilibrium_solutions[:, i])
            if try_previous_converged:
                u_tries.append(u_prev)
            # Append heuristic/element-based guesses
            u_tries.extend(self.get_initial_dof_guess(p) or [])

            success = False
            sol = None
            for j, u_init in enumerate(u_tries):
                # if verbose:
                #     print(f'Theta {np.degrees(t):.2f} trying with u guess {j}')
                sol, success = self.solve_equilibrium(u_init, p)
                if success:
                    if verbose:
                        print(f'Theta {np.degrees(t):.0f}°: converged with initial guess #{j}')
                    equilibrium_solutions[:, i] = sol
                    u_prev = sol
                    try_previous_converged = True
                    break

            if not success and verbose:
                # print(f'Theta {np.degrees(t):.0f}°: did not converge with provided guesses')# print(f'Theta {np.degrees(t):.0f} UNCONVERGED with initial u guess {j}')
                pass

            converged.append(bool(success))

        # 6) Secondary Analysis Attempt to "Fill-in" Failed Convergence Points via Interpolation
        while any(converged) and not all(converged):
            if verbose:
                print('Attempting to reanalyze unconverged points...')
            unconverged_idx = [i for i, suc in enumerate(converged) if not suc]
            new_converge_found = False
            for idx in unconverged_idx:
                t = theta_z[idx]
                p = loads[:,idx]
                u_tries = self.get_interpolated_dof_guess(idx, converged, equilibrium_solutions)
                success = False
                sol = None
                j = 0
                for j, u_init in enumerate(u_tries):
                    sol, success = self.solve_equilibrium(u_init, p)
                    if success:
                        if verbose:
                            print(f'Theta {np.degrees(t):.0f}°: converged with interpolated guess #{j}')
                        equilibrium_solutions[:, idx] = sol
                        new_converge_found = True
                        converged[idx] = success
                        break


                if not success and verbose:
                    print(f'Theta {np.degrees(t):.0f} UNCONVERGED with interpolated u guess {j}')
            if not new_converge_found:
                break

        # Compile arrays of element forces
        if not any(converged):
            raise Exception(
                f'Item {self.equipment_info.equipment_id} failed to converge to any solutions. Check geometry definition.')

        return Solution(
            theta_z=theta_z,
            equilibrium_solutions=equilibrium_solutions,
            converged=converged,
            factor_method=factor_method
        )
        # todo: handle post-processing of evaluating all elements
        # self.get_element_results()

    def get_load_vector(self, fh, fv, theta_z: NDArray) -> NDArray:
        eq = self.equipment_props
        n_dof = self.dof_props.n_dof

        vx = fh * np.cos(theta_z)
        vy = fh * np.sin(theta_z)
        fz = np.full_like(theta_z, fv)
        mx = -vy * eq.zCG + fz * eq.ey
        my = vx * eq.zCG - fz * eq.ex
        t = -vx * eq.ey + vy * eq.ex

        p6 = np.vstack([vx, vy, fz, mx, my, t])  # (6, n_theta)

        if n_dof == 6:
            return p6
        else:
            n_theta = theta_z.shape[0]
            p = np.zeros((n_dof, n_theta))
            p[:6, :] = p6
            return p

    def get_initial_dof_guess(self,p):
        """ Returns an initial guess for the DOF displacements U0 = [dx, dy, dz, rx, ry, rz, ...]"""

        u_init_list = [self._center_of_rigidity_based_dof_guess(p, rot_magnitude=0.01/min(self.equipment_props.Bx,
                                                                                          self.equipment_props.By)),# Initial guess at given angle
                       self._center_of_rigidity_based_dof_guess(p, rot_magnitude=1e-7),
                       self._center_of_rigidity_based_dof_guess(p + np.pi / 16),  # At perturbed angle
                       self._center_of_rigidity_based_dof_guess(p - np.pi / 16)]
        return u_init_list

    def _center_of_rigidity_based_dof_guess(self,p, disp_magnitude=1e-6, rot_magnitude=1e-6):
        eprops = self.equipment_props
        u = np.zeros(self.dof_props.n_dof)

        # Calculate Moments about COR
        fx, fy, fz = p[0:3]

        sign_fx = 1 if fx >= 0 else -1
        sign_fy = 1 if fy >= 0 else -1
        sign_fz = 1 if fz >= 0 else -1

        cor_x = self.analysis_vars.cor_coordinates['x_displacement'][sign_fx]  # COR for displacement in x-direction
        cor_y = self.analysis_vars.cor_coordinates['y_displacement'][sign_fy]  # COR for displacement in y-direction
        cor_z = self.analysis_vars.cor_coordinates['z_displacement'][sign_fz]  # COR for displacement in z-direction

        mx = (cor_y[2] - eprops.zCG) * fy - (cor_z[1] - eprops.ey) * fz
        my = -(cor_x[2] - eprops.zCG) * fx + (cor_z[0] - eprops.ex) * fz
        mz = (cor_x[1] - eprops.ey) * fx - (cor_y[0] - eprops.ex) * fy
        m_array = np.array((mx, my, mz))

        # Apply DOF Rotations proportional to moments about COR
        unit_rot = m_array / np.linalg.norm(m_array)
        rx, ry, rz = rot_magnitude * unit_rot

        # Apply DOF translations are proportional to applied forces such that rotation occurs about COR
        unit_disp = p[0:3] if p[0:3].sum() == 0 else p[0:3] / np.linalg.norm(p[0:3])
        cor_disp = disp_magnitude * unit_disp

        y, z = cor_x[1:]
        dx = cor_disp[0] -z * ry + y * rz

        x = cor_y[0]
        z = cor_y[2]
        dy =  cor_disp[1] + z * rx - x * rz

        x = cor_z[0]
        y = cor_z[1]
        dz = cor_disp[2] + x * ry - y * rx

        u[0:6] = [dx, dy, dz, rx, ry, rz]

        # Compute Floor Plate DOF initial guesses
        for dof_map, plate in zip(self.dof_props.base_plate_dofs, self.elements.base_plates):

            # # NEW METHOD
            # if len(plate.free_dofs)>0:
            #     u_free, success = plate.solve_released_dof_equilibrium(u, dof_map)
            #     if success:
            #         u[dof_map[plate.free_dofs]] = u_free


            ## OLD METHOD
            dzp = dz + plate.props.yc * rx - plate.props.xc * ry  # z-displacement at (xc,yc)

            # # Handle Vertical Releases
            # if (plate.release_zp and not plate.release_zn and dzp >=0) or (plate.release_zn and not plate.release_zp and dzp <0):
            #     if plate.release_mx:
            #         u[dof_map[3]] = 0
            #     if plate.release_my:
            #         u[dof_map[4]] = 0
            #     return u

            # Handle Independent Vertical DOF (Triggered when both zn and zp releases are present)
            if plate.props.releases.zp and plate.props.releases.zn:
                # dzp = max(0, 0.5*dzp)
                dzp = 0.5 * abs(dzp)
                # dzp = 1e-7
                u[dof_map[2]] = dzp

            dza = 0.5 * abs(dzp)  # Initial anchor point "small tension" target value

            # Handle Rotation Releases
            if plate.props.releases.mx and plate.props.releases.my:
                dx = plate.props.xy_anchors[:, 0] - plate.props.x0
                dy = plate.props.xy_anchors[:, 1] - plate.props.y0
                delta = dza - dzp

                # Safe division: skip entries with zero denominator
                with np.errstate(divide='ignore', invalid='ignore'):
                    ryp = np.divide(-0.5 * delta, dx, where=dx != 0)
                    rxp = np.divide(0.5 * delta, dy, where=dy != 0)

                # Filter out any NaNs that might remain due to zero division
                ryp = ryp[~np.isnan(ryp)]
                rxp = rxp[~np.isnan(rxp)]

                if rxp.size > 0:
                    u[dof_map[3]] = np.mean(rxp)
                if ryp.size > 0:
                    u[dof_map[4]] = np.mean(ryp)

            elif plate.props.releases.mx and not plate.props.releases.my:
                dy = plate.props.xy_anchors[:, 1] - plate.props.y0
                dx = plate.props.xy_anchors[:, 0] - plate.props.x0
                delta = dza - dzp + ry * dx

                with np.errstate(divide='ignore', invalid='ignore'):
                    rxp = np.divide(delta, dy, where=dy != 0)

                rxp = rxp[~np.isnan(rxp)]
                if rxp.size > 0:
                    u[dof_map[3]] = np.mean(rxp)

            elif plate.props.releases.my and not plate.props.releases.mx:
                dy = plate.props.xy_anchors[:, 1] - plate.props.y0
                dx = plate.props.xy_anchors[:, 0] - plate.props.x0
                delta = dza - dzp - rx * dy

                with np.errstate(divide='ignore', invalid='ignore'):
                    ryp = np.divide(delta, -dx, where=dx != 0)

                ryp = ryp[~np.isnan(ryp)]
                if ryp.size > 0:
                    u[dof_map[4]] = np.mean(ryp)
        return u

    @staticmethod
    def get_interpolated_dof_guess(idx, converged, equilibrium_solutions):
        """ Returns trial solutions for an unconverged point
        by taking weighted averages of bounding converged solutions points"""

        if not any(converged):
            raise Exception("No converged solutions with which to interpolate initial guess.")

        sol_before = None
        sol_after = None

        n = len(converged)

        # Wrapped search backward
        for offset in range(1, n):
            i = (idx - offset) % n
            if converged[i]:
                sol_before = equilibrium_solutions[:, i]
                break

        # Wrapped search forward
        for offset in range(1, n):
            i = (idx + offset) % n
            if converged[i]:
                sol_after = equilibrium_solutions[:, i]
                break

        u_tries = [0.5 * sol_before + 0.5 * sol_after,
                   0.25 * sol_before + 0.75 * sol_after,
                   0.75 * sol_before + 0.25 * sol_after,
                   sol_after]

        return u_tries

    def solve_equilibrium(self, u_init, p):
        methods = ['hybr']
        for method in methods:
            self.analysis_vars.u_previous = None
            self.analysis_vars.residual_call_count = 0
            res = root(self.equilibrium_residual, u_init, args=p, method=method, #xtol=1e-8,
                       options={'maxfev': 200,'xtol':1e-6})
            if res.success:
                # print(f'Success with {res.nfev} function calls')
                break
        sol = res.x
        return sol, res.success

    def equilibrium_residual(self, u, p, stiffness_update_threshold=0.01, penalty_factor=1e3, verbose=False):
        """Returns the residual for the equilibrium equation p = ku
        stiffness_update_threshold indicates percentage of dof norm change at which stiffness matrix should be updated.
        penalty factor is applied to zero-force dof residuals to ensure better convergence"""
        n_dof = self.dof_props.n_dof
        load_vector_weights = np.ones(n_dof)
        load_vector_weights += 20 * np.array(self.dof_props.disp_dofs)

        disp_vector_weights = np.ones(n_dof)
        # disp_vector_weights[0:3] = [0.1, 0.1, 0.1]

        # Update Stiffness Matrix for large change in norm of u or if any base anchors have reversed signs
        if self.analysis_vars.K is None or self.analysis_vars.u_previous is None:
            update_K = True  # First iteration, always update K
        else:
            norm_change = np.linalg.norm(disp_vector_weights * u - disp_vector_weights * self.analysis_vars.u_previous) / \
                          (np.linalg.norm(disp_vector_weights * self.analysis_vars.u_previous) + 1e-12)
            norm_change_trigger = norm_change > stiffness_update_threshold

            # num_base_anchors_in_tension = 0 if self.base_anchors is None else sum(
            #     Utilities.vertical_point_displacements(self.base_anchors.xy_anchors, u) > 0)
            # anchor_change_trigger = num_base_anchors_in_tension != self._num_base_anchors_in_tension

            update_K = norm_change_trigger  # or anchor_change_trigger
            # self._num_base_anchors_in_tension = num_base_anchors_in_tension.copy()

        self.analysis_vars.residual_call_count+=1
        if self.analysis_vars.residual_call_count % 10 == 0:
            update_K = True

        if update_K:
            self.analysis_vars.K = self.update_stiffness_matrix(u)
            self.analysis_vars.u_previous = u.copy()

        # Penalty to prevent "run-away" dofs
        disp_limit = 10
        rotation_limit = 1

        disp_indices = np.where(np.array(self.dof_props.disp_dofs) == 1)[0]
        rot_indices = np.where(np.array(self.dof_props.disp_dofs) == 0)[0]

        # Extract corresponding u values
        disp_values = u[disp_indices]
        rot_values = u[rot_indices]

        # Apply penalty logic
        disp_excess = np.maximum(0, np.abs(disp_values) - disp_limit)
        rot_excess = np.maximum(0, np.abs(rot_values) - rotation_limit)

        # Define a quadratic penalty for each DOF
        penalty_disp = 1e6 * disp_excess ** 2
        penalty_rot = 1e6 * rot_excess ** 2
        penalty_vector = np.zeros(n_dof)
        penalty_vector[disp_indices] = penalty_disp
        penalty_vector[rot_indices] = penalty_rot

        # Penalty on zero-force dofs
        zero_force_mask = np.isclose(p, 0, 1e-10)
        load_vector_weights[zero_force_mask] *= penalty_factor
        residual = load_vector_weights * (-np.dot(self.analysis_vars.K, u) - p) + penalty_vector
        if verbose:
            print(f'Norm: {np.linalg.norm(residual):.3e}, Update K: {update_K}, Penalties: {penalty_vector}')
        return residual

    def evaluate(self, solutions: dict[FactorMethod, Solution]) -> ElementResults:

        base_plate_results = []
        base_anchor_results = []
        base_cxn_results = []
        base_sms_results = []
        wall_bracket_results = []
        wall_backing_results = []
        bracket_cxn_results = []
        bracket_sms_results = []
        wall_anchor_results = []

        for factor_method, solution in solutions.items():
            dof_sol = solution.equilibrium_solutions

            # Base Plates (Downstream elements, evaluate for all methods)
            bp_results = [plate.evaluate(dof_sol) for plate in self.elements.base_plates]
            if self.elements.base_plates[0].factor_method == factor_method:
                base_plate_results = bp_results.copy()

            # Base Anchors (No downstream elements, only evaluate for applicable method)
            for anch_idx, anchors in enumerate(self.elements.base_anchors):
                if self.elements.base_anchors[0].factor_method == factor_method:
                    bp_indices = self.elements.anchors_to_bp[anch_idx]
                    # Pull anchor forces from all contributing baseplates
                    anchor_forces = np.concatenate([bp_results[bp_idx].anchor_forces for bp_idx in bp_indices],axis=0)
                    base_anchor_results.append(anchors.evaluate(anchor_forces))

            # Base Connections (Only Evaluate on sms-required method)
            for bp, cxn_idx in zip(bp_results, self.elements.bp_to_cxn):
                if cxn_idx is None:
                    continue
                if self.elements.base_plate_connections[0].factor_method == factor_method:
                    bp_cxn = self.elements.base_plate_connections[cxn_idx]
                    cxn_results = bp_cxn.evaluate(bp.connection_forces)
                    base_cxn_results.append(cxn_results)

                    bp_sms = self.elements.base_plate_fasteners[self.elements.cxn_to_sms[cxn_idx]]
                    base_sms_results.append(bp_sms.evaluate(cxn_results.anchor_forces_loc))

            # Brackets (Downstream elements, evaluate for all methods, return results for applicable method)
            bracket_results = [bracket.evaluate(dof_sol,factor_method) for bracket in self.elements.wall_brackets]
            if self.elements.wall_brackets[0].factor_method == factor_method:
                wall_bracket_results = bracket_results.copy()

            # Wall Backing and Anchors
            if self.elements.wall_anchors and self.elements.wall_anchors[0].factor_method==factor_method:
                for backing, indices in zip(self.elements.wall_backing, self.elements.backing_to_brackets):
                    bracket_forces = np.stack([bracket_results[idx].reactions_backing for idx in indices],axis=0)
                    wall_backing_results.append(backing.evaluate(bracket_forces))
                for anchors, indices in zip(self.elements.wall_anchors, self.elements.anchors_to_backing):
                    anchor_forces = np.concatenate([wall_backing_results[idx].anchor_forces_loc for idx in indices],axis=0)
                    wall_anchor_results.append(anchors.evaluate(anchor_forces))

            # Bracket Connections
            if self.elements.wall_bracket_connections and \
                self.elements.wall_bracket_fasteners[0].factor_method == factor_method:

                for bracket, bracket_cxn, bracket_sms in zip(
                        bracket_results, self.elements.wall_bracket_connections, self.elements.wall_bracket_fasteners):
                    cxn_results = bracket_cxn.evaluate(bracket.reactions_equipment)
                    bracket_cxn_results.append(cxn_results)
                    bracket_sms_results.append(bracket_sms.evaluate(cxn_results.anchor_forces_loc))

        return ElementResults(
            base_plates = base_plate_results,
            base_anchors = base_anchor_results,
            base_plate_connections = base_cxn_results,
            base_plate_fasteners = base_sms_results,
            wall_brackets=wall_bracket_results,
            wall_backing=wall_backing_results,
            wall_bracket_connections=bracket_cxn_results,
            wall_bracket_fasteners=bracket_sms_results,
            wall_anchors=wall_anchor_results
        )


