from asyncio import Protocol
from dataclasses import dataclass, fields
from typing import Optional, Union, List, Tuple, Literal
import numpy as np
from numpy.typing import NDArray
from enum import Enum

import anchor_pro.ap_types
from anchor_pro.ap_types import WallPositions
from anchor_pro.utilities import get_anchor_spacing_matrix

'''ENUMERATION DEFINITIONS'''
# Enumerations
class Profiles(str, Enum):
    slab = 'Slab'
    wall = 'Wall'
    deck = 'Filled Deck'

class AnchorPosition(str, Enum):
    top = 'Top'
    soffit = 'Soffit'

class InstallationMethod(str, Enum):
    post = 'Post-Installed'
    cast = 'Cast-in-Place'

class AnchorTypes(str, Enum):
    expansion = 'Expansion'
    screw = 'Screw'
    threaded = 'Threaded Rod'
    headed_stud = 'Headed Stud'  # ACI 17.6.2.2.3, Headed anchors
    headed_bolt = 'Headed Bolt'
    adhesive = 'Adhesive'

class ShearDirLabel(str, Enum):
    XP = 'X+'
    XN = 'X-'
    YP = 'Y+'
    YN = 'Y-'

class PerpToEdgeIdx(int, Enum):
    # Index of xy_anchors for coordinate along axis perpendicular to edge
    XP = 0
    XN = 0
    YP = 1
    YN = 1

class ParToEdgeIdx(int, Enum):
    # Index of xy_anchors for coordinate along axis parallel to edge
    XP = 1
    XN = 1
    YP = 0
    YN = 0

'''DATA CLASSES'''
@dataclass(frozen=True, slots=True)
class EdgeOrdinates:
    XP: Union[float,np.inf]
    XN: Union[float, np.inf]
    YP: Union[float, np.inf]
    YN: Union[float, np.inf]

class AnchorGroup(Protocol):
    anchor_indices: List[int]


@dataclass(frozen=True, slots=True)
class TensionGroup:
    #group_idx: int
    anchor_indices: List[int]  # shape (n_g,)
    n_anchor: int
    centroid: NDArray  # shape (2,)

    # Anchor edge distances
    cax_neg: float
    cax_pos: float
    cay_neg: float
    cay_pos: float

    ca_min: float

    # Anchor spacing
    sx: dict   #todo [CODE QA]: review how spacing dict are used and simplify as appropriate
    sy: dict



@dataclass(frozen=True,slots=True)
class ShearGroup:
    # group_idx: int
    anchor_indices: list[int]  # shape (n_g,)
    n_anchor: int
    centroid: NDArray  # shape (2,)
    direction: ShearDirLabel

    ca1: float  # distance perpendicular to breakout edge
    ca2p: float  # distance parallel to breakout edge
    ca2n: float
    ca2p_ordinate: float  # Ordinate to side ege measured on axis parallel to breakout edge
    ca2n_ordinate: float


    # s_total: float  # total spacing between anchors
    s_max: float  # maximum spacing between anchors

@dataclass(frozen=True, slots=True)
class GeoProps:
    xy_anchors: NDArray  # shape (n_anchor, 2)
    Bx: float
    By: float
    cx_neg: Union[float,np.inf]  # Edge of concrete relative to bounding box (Bx and By)
    cx_pos: Union[float,np.inf]
    cy_neg: Union[float,np.inf]
    cy_pos: Union[float,np.inf]

    anchor_position: AnchorPosition = AnchorPosition.top

    # Post Processed Attributes
    n_anchor: Optional[int] = None
    c_min: Optional[float] = None  # Minimum provided edge distance
    s_min: Optional[float] = None  # Minimum provided anchor spacing
    edge_ordinates: Optional[EdgeOrdinates] = None
    tension_groups_matrix: Optional[NDArray] = None  # (n_anchor, n_anchor)
    shear_groups_map: List[List[int]] = None
    tension_groups: List[TensionGroup] = None
    shear_groups: List[ShearGroup] = None
    supporting_wall: Optional[WallPositions] = None
    def __post_init__(self):
        # Number of Anchors
        n_anchor = self.xy_anchors.shape[0]

        #Define Edge Ordinates
        XP = 0.5 * self.Bx + self.cx_pos
        XN = -0.5 * self.Bx - self.cx_neg
        YP = 0.5 * self.By + self.cy_pos
        YN = -0.5 * self.By - self.cy_neg

        edges = EdgeOrdinates(
            XP=XP if np.isfinite(XP) else np.inf,
            XN=XN if np.isfinite(XN) else np.inf,
            YP=YP if np.isfinite(YP) else np.inf,
            YN=YN if np.isfinite(YN) else np.inf
        )

        x = self.xy_anchors[:, 0]
        y = self.xy_anchors[:, 1]

        # Distances of each point to each edge (n, 4)
        D = np.stack([
            np.abs(x - XP),  # to x = XP (right)
            np.abs(x - XN),  # to x = XN (left)
            np.abs(y - YP),  # to y = YP (top)
            np.abs(y - YN),  # to y = YN (bottom)
        ], axis=1)
        c_min = D.min()

        spacing_matrix = get_anchor_spacing_matrix(self.xy_anchors)
        idx = np.triu_indices_from(spacing_matrix, k=1)
        s_min = spacing_matrix[idx].min()

        # Set Attributes
        object.__setattr__(self,'n_anchor',n_anchor)
        object.__setattr__(self, 'edge_ordinates', edges)
        object.__setattr__(self, 'c_min', c_min)
        object.__setattr__(self, 's_min', s_min)

@dataclass(frozen=True, slots=True)
class AnchorBasicInfo:
    anchor_id: str
    installation_method: InstallationMethod
    anchor_type: AnchorTypes
    manufacturer: str
    product: str
    product_type: str
    esr: str
    cost_rank: float

@dataclass(frozen=True, slots=True)
class Phi:
    saN: float  # Steel Tension
    pN: float  # Pullout Tension
    cN: float  # Concrete Breakout Tension
    cV: float  # Concrete Breakout Shear
    saV: float  # Steel Shear
    cpV: float  # Concrete Pryout Shear
    eqV: float  # Steel Seismic Shear
    eqN: float  # Steel Seismic Tension
    seismic: float  # Seismic
    aN: Optional[float] = None # Adhesive Bond

@dataclass(frozen=True, slots=True)
class ConcreteProps:
    weight_classification: Literal["LWC","NWC"]
    profile: Profiles
    fc: float

    lw_factor: float

    cracked_concrete: bool
    poisson: float
    t_slab: float  # can default to np.inf at instantiation
    lw_adjustment: float = 0.8  # See Table 17.2.4.1 (Default might be 0.8)
    lw_bond_adjustment: float = 0.6  # See Table 17.2.4.1 (Default might be 0.6)
    lw_factor_a: float = None
    lw_factor_bond_failure: float = None

    def __post_init__(self):
        if self.weight_classification == 'LWC':
            lw_factor_a = self.lw_factor * self.lw_adjustment
            lw_factor_bond_failure = self.lw_factor * self.lw_bond_adjustment
        else:
            lw_factor_a = self.lw_factor
            lw_factor_bond_failure = self.lw_factor

        object.__setattr__(self, 'lw_factor_a', lw_factor_a)
        object.__setattr__(self, 'lw_factor_bond_failure', lw_factor_bond_failure)
@dataclass(frozen=True, slots=True)
class MechanicalAnchorProps:
    info: AnchorBasicInfo
    fya: float
    fua: float
    Nsa: float
    Np: float
    kc: float  # ACI318-19 17.6.2.2.1: 24 for cast-in, 17 for post installed, or per mfr
    kc_uncr: float
    kc_cr: float
    le: float  # 17.7.2.2
    da: float
    cac: float
    esr: str
    hef_default: float
    Vsa: float
    K: float
    K_cr: float
    K_uncr: float
    Kv: float  # Shear Stiffness
    hmin: float  # Minimum permitted member thickness required
    c1: float  # Minimum permitted edge distances and spacing
    s1: float
    c2: float
    s2: float
    phi: Phi
    abrg: Optional[float] = None  # Net bearing area of headed stud or anchor

@dataclass(frozen=True, slots=True)
class AdhesiveAnchorProps:
    info: AnchorBasicInfo
    esr_adhesive: str
    bar_id: str
    type: str
    standard: str
    spec: str
    grade: str
    da: float
    da_inside: float
    Ase: float
    Ase_inside: float
    Nsa: float
    Nsa_inside: float  # todo [ADHESIVE]: implement use
    Vsa: float
    alpha_Vseis: float  # todo [ADHESIVE]: implement use.
    alpha_Nseis: float
    kc: float
    hef_min: float
    hef_max: float
    hef_default: float
    hmin: float
    s_min: float
    c_min: float
    cc_min: float  # Required clear cover
    max_torque: float
    tau: float
    category: str
    exp_bond: float  # exponent for strength amplification formula (fc / fc_min) ^ exp_bond
    maxtemp_A_short: float
    maxtemp_B_short: float
    maxtemp_long: float
    fc_min: float
    fc_max: float

    # Cure time tables
    t_work_23: float
    t_init_23: float
    t_full_23: float
    t_work_32: float
    t_init_32: float
    t_full_32: float
    t_work_40: float
    t_init_40: float
    t_full_40: float
    t_work_41: float
    t_init_41: float
    t_full_41: float
    t_work_50: float
    t_init_50: float
    t_full_50: float
    t_work_60: float
    t_init_60: float
    t_full_60: float
    t_work_68: float
    t_init_68: float
    t_full_68: float
    t_work_70: float
    t_init_70: float
    t_full_70: float
    t_work_72: float
    t_init_72: float
    t_full_72: float
    t_work_85: float
    t_init_85: float
    t_full_85: float
    t_work_86: float
    t_init_86: float
    t_full_86: float
    t_work_90: float
    t_init_90: float
    t_full_90: float
    t_work_95: float
    t_init_95: float
    t_full_95: float
    t_work_100: float
    t_init_100: float
    t_full_100: float
    t_work_104: float
    t_init_104: float
    t_full_104: float
    t_work_105: float
    t_init_105: float
    t_full_105: float
    t_work_110: float
    t_init_110: float
    t_full_110: float
    phi: Phi

@dataclass(frozen=True,slots=True)
class SpacingRequirements:
    edge_and_spacing_ok: bool
    slab_thickness_ok: bool
    ok: bool

@dataclass(frozen=True, slots=True)
class SteelTensionCalc:
    demand: NDArray
    phi: float
    phi_seismic: float
    Nsa: float
    unities: NDArray

@dataclass(frozen=True,slots=True)
class TensionBreakoutCalc:
    hef_default: float
    hef_limit: float
    hef: float
    bxN: float
    byN: float
    Anc: float
    Anco: float
    Nb: float
    Ncb: float
    ex: NDArray[np.int64]  #(n_theta,)
    ey: NDArray[np.int64]  #(n_theta,)
    psi_ecNx: NDArray[np.int64]  #(n_theta,)
    psi_ecNy: NDArray[np.int64]
    psi_ecN: NDArray[np.int64]
    psi_edN: float
    psi_cN: float
    psi_cpN: float
    phi: float
    phi_seismic:float
    demand: NDArray[np.int64]
    unities: NDArray[np.int64]

@dataclass(frozen=True, slots=True)
class TensionPulloutCalc:
    Np: float
    demand: NDArray
    phi: float
    phi_seismic: float
    unities: NDArray

@dataclass(frozen=True, slots=True)
class BlowoutCalc:
    Nsb: float
    Nsbg: float
    demand: NDArray
    phi: float
    phi_seismic: float
    unities: NDArray

@dataclass(frozen=True, slots=True)
class BondStrengthCalc:
    cna: float
    tau: float
    ANao: float
    Ana: float
    Nba: float
    Na: float
    Nag: float
    bxN: float
    byNa: float
    psi_ecNa: float
    psi_edNa: float
    psi_cpNa: float
    demand: NDArray
    phi: float
    phi_seismic: float
    unities: NDArray

@dataclass(frozen=True, slots=True)
class SteelShearCalc:
    demand: NDArray
    phi: float
    phi_seismic: float
    Vsa: float
    unities: NDArray

@dataclass(frozen=True, slots=True)
class ShearBreakoutCalc:
    ca1_eff: float
    # ca2p_eff: float
    # ca2n_eff: float
    Avco: float
    b: float
    ha_eff: float
    Avc: float
    ecc: NDArray
    psi_ecV: NDArray
    psi_edV: float
    psi_cV: float
    psi_hV: float
    Vb: float
    Vcb: NDArray
    demand: NDArray
    phi: float
    phi_seismic: float
    unities: NDArray


@dataclass(frozen=True, slots=True)
class PryoutCalc:
    demand: NDArray
    phi: float
    phi_seismic: float
    kcp: float
    Vcp: float
    unities: NDArray

@dataclass(frozen=True, slots=True)
class ConcreteAnchorResults:
    tension_groups: List[TensionGroup]
    shear_groups: List[ShearGroup]
    spacing_requirements: SpacingRequirements
    steel_tension_calcs: List[SteelTensionCalc]
    tension_breakout_calcs: List[TensionBreakoutCalc]
    anchor_pullout_calcs: List[TensionPulloutCalc]
    side_face_blowout_calcs: List[BlowoutCalc]
    bond_strength_calcs: List[BondStrengthCalc]
    steel_shear_calcs: List[SteelShearCalc]
    shear_breakout_calcs: List[ShearBreakoutCalc]
    shear_pryout_calcs: List[PryoutCalc]

    tension_unity_by_anchor: NDArray #(n_anchor, n_theta)
    shear_unity_by_anchor: NDArray #(n_anchor, n_theta)
    unity_by_anchor: NDArray  #(n_anchor, n_theta)

    unity: float
    ok: bool
    governing_anchor_idx: int
    governing_theta_idx: int

    governing_tension: float
    governing_shear: float
    governing_tension_group: int
    governing_shear_group: int

    forces: NDArray #(n_anchor, 3, n_theta)
    K: float
    cost_rank: float



SupportedConcreteAnchors = Union[MechanicalAnchorProps,AdhesiveAnchorProps]

def _fill_group_matrix(boolean_matrix: NDArray[bool])->NDArray[np.bool_]:
    """Input assumed to be (n_anchor,n_anchor) boolean matrix of in_group condition where anchor j is in group with i
    This function will iterate over the matrix and return a group matrix such that
    if B is in group is A and C is in group with B then B and C are ingroup with A.
    Computes transitive closure under undirected proximity"""

    boolean_matrix = boolean_matrix.astype(int)  # Convert the boolean matrix to integer for matrix operations
    groups_matrix = np.copy(boolean_matrix)

    # Keep multiplying until there are no new connections found
    while True:
        new_matrix = groups_matrix @ boolean_matrix

        # Ensure we don't exceed the boolean logic (1 is True, everything above 1 is still True)
        new_matrix[new_matrix > 1] = 1

        # Check if the matrix has changed
        if np.array_equal(new_matrix, groups_matrix):
            break
        else:
            groups_matrix = new_matrix

    # Convert back to boolean
    return groups_matrix.astype(bool)

def _get_tension_group_indices(groups_matrix: NDArray[bool]) ->List[List[int]]:
    # Extract Group Anchor indices
    n = groups_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    groups = []

    for i in range(n):
        if not visited[i]:
            group_indices = np.where(groups_matrix[i])[0]
            groups.append(group_indices.tolist())
            visited[group_indices] = True  # Mark all in group as visited

    return groups

def _get_shear_group_indices(groups_matrix: NDArray[bool]) ->List[List[int]]:
    # Extract Group Anchor indices
    n = groups_matrix.shape[0]
    groups = set()

    for i in range(n):
            group_indices = np.where(groups_matrix[i])[0]
            groups.add(tuple(group_indices))
    return [list(g) for g in groups]

def get_tension_groups(radius: float, xy_anchors: NDArray, edge_ordinates: EdgeOrdinates) -> List[TensionGroup]:
    spacing_matrix = get_anchor_spacing_matrix(xy_anchors)
    boolean_matrix = spacing_matrix <= radius
    groups_matrix = _fill_group_matrix(boolean_matrix)
    group_indices_lists = _get_tension_group_indices(groups_matrix)
    tension_groups = []
    for indices in group_indices_lists:
        cax_pos = abs(edge_ordinates.XP - np.max(xy_anchors[indices, 0]))
        cax_neg = abs(edge_ordinates.XN - np.min(xy_anchors[indices, 0]))
        cay_pos = abs(edge_ordinates.YP - np.max(xy_anchors[indices, 1]))
        cay_neg = abs(edge_ordinates.YN - np.min(xy_anchors[indices, 1]))

        # Define Spacing Dictionaries
        #todo [CODE QA]: review how spacing dict used and simplify as appropriate
        sy = {}
        sx = {}
        xy_group = xy_anchors[indices,:]

        idx = np.lexsort((xy_group[:, 0], xy_group[:, 1]))
        sorted_by_x = xy_group[idx]

        idx = np.lexsort((xy_group[:, 1], xy_group[:, 0]))
        sorted_by_y = xy_group[idx]

        # Calculate spacing in y-direction
        for x in np.unique(sorted_by_x[:, 0]):
            pts = sorted_by_x[sorted_by_x[:, 0] == x]
            if len(pts) > 1:
                sy[x] = np.diff(pts[:, 1])
            else:
                sy[x] = np.array([0])

        # Calculate spacing in x-direction
        for y in np.unique(sorted_by_y[:, 1]):
            pts = sorted_by_y[sorted_by_y[:, 1] == y]
            if len(pts) > 1:
                sx[y] = np.diff(pts[:, 0])
            else:
                sx[y] = np.array([0])

        tension_groups.append(
            TensionGroup(
                anchor_indices=indices,
                n_anchor=len(indices),
                centroid = np.mean(xy_anchors[indices],axis=0),
                cax_pos = cax_pos,
                cax_neg = cax_neg,
                cay_pos = cay_pos,
                cay_neg = cay_neg,
                ca_min = min(cax_pos,cax_neg,cay_pos,cay_neg),
                sx=sx,
                sy=sy
            )
        )

    return tension_groups


def get_shear_groups(xy_anchors: NDArray, edge_ordinates: EdgeOrdinates) -> List[ShearGroup]:
    shear_groups = []
    for f in fields(edge_ordinates):
        direction = f.name
        # Calc all edge distances
        edge1 = getattr(edge_ordinates,direction)
        if np.isinf(edge1): # Skip directions with infinite edge distances
            continue
        if direction == 'XP' or direction == 'XN':
            edge2p = edge_ordinates.YP
            edge2n = edge_ordinates.YN
        else:
            edge2p = edge_ordinates.XP
            edge2n = edge_ordinates.XN

        perp_idx = PerpToEdgeIdx[direction]
        par_idx = ParToEdgeIdx[direction]
        ca1_array = abs(edge1 - xy_anchors[:,perp_idx])  #(n_anchor,)

        # Calc provided anchor spacing
        spacing_parallel_to_edge = abs(xy_anchors[:,None,:]-xy_anchors[None,:,:])[:,:,par_idx]

        # Calc is_group spacing (1.5*c11 + 1.5*c12)
        max_in_group_spacing = 1.5*(ca1_array[:,None] + ca1_array[None,:])

        # Identify groups G s.t. Gij = True if anchor i and j overlap breakout cones and j is not further from the edge than i
        is_not_behind = ca1_array[:, None] >= ca1_array[None, :]  # exclude from group any anchors further from the edge than the current anchor
        boolean_matrix = (spacing_parallel_to_edge < max_in_group_spacing) & is_not_behind
        groups_matrix = _fill_group_matrix(boolean_matrix)
        group_indices_lists = _get_shear_group_indices(groups_matrix)

        for indices in group_indices_lists:
            ca1_group = ca1_array[indices]
            ca1 = max(ca1_group)
            xy_group = xy_anchors[indices]

            ca2p_ordinate = min(np.max(xy_group[:,par_idx] + 1.5*ca1_group),edge2p)
            ca2n_ordinate = max(np.min(xy_group[:,par_idx] - 1.5*ca1_group), edge2n)

            ca2p = abs(ca2p_ordinate - np.max(xy_group[:,par_idx]))
            ca2n = abs(ca2n_ordinate - np.min(xy_group[:, par_idx]))

            s_group = spacing_parallel_to_edge[np.ix_(indices, indices)]
            s_max = np.max(np.min(np.where(s_group==0,np.inf,s_group)))
            if not np.isfinite(s_max):
                s_max = 0

            shear_groups.append(
                ShearGroup(
                    anchor_indices=indices,
                    n_anchor = len(indices),
                    centroid = np.mean(xy_anchors[indices,:],axis=0),
                    direction = ShearDirLabel[direction],
                    ca1 = ca1,
                    ca2p_ordinate = ca2p_ordinate,
                    ca2n_ordinate = ca2n_ordinate,
                    ca2p = ca2p,
                    ca2n = ca2n,
                    s_max = s_max
                )
            )
    return shear_groups

def _shear_resultant(forces,x_idx=0, y_idx=1):
    n_dim = np.ndim(forces)
    if n_dim < 3:
        raise Exception(f"Expected forces array to be dim=3 (n_anchor,3,n_theta). Got ndim = {n_dim}")
    if y_idx != x_idx+1:
        raise Exception("x and y indices invalid. Must be two consecutive columns (i.e. 1 and 2)")
    return np.sum(np.linalg.norm(forces[:,x_idx:(y_idx+1),:],axis=1),axis=0)

def group_eccentricity(xy_group: NDArray[np.float64],
                                 forces: NDArray
                                 )->Tuple[NDArray[np.float64],NDArray[np.float64]]:
    """
    xy_group: (n_g,2) Array of anchor coordinates
    tensions: (n_g, n_theta) array of tension forces on group anchors
    """
    w = forces.copy()
    zero_cols = np.all(w == 0.0, axis=0)  # (n_theta,)
    # For zero columns, use equal weights; this makes the weighted centroid = mean centroid
    w[:, zero_cols] = 1.0
    x = xy_group[:, 0][:, None]  # (n_g, 1)
    y = xy_group[:, 1][:, None]  # (n_g, 1)
    sumw = w.sum(axis=0)  # (n_theta,)
    x_w = (x * w).sum(axis=0) / sumw  # (n_theta,)
    y_w = (y * w).sum(axis=0) / sumw

    # Group centroid (unweighted); if tg.centroid is already this, you can use it
    x_c = float(xy_group[:, 0].mean())
    y_c = float(xy_group[:, 1].mean())

    ex = x_w - x_c
    ey = y_w - y_c

    ex = np.where(zero_cols, 0.0, ex)
    ey = np.where(zero_cols, 0.0, ey)

    return ex, ey

def reshape_unities_array(unities: np.ndarray) -> np.ndarray:
    """Ensures unities array is (n_group, n_theta) even if n_group = 1"""
    import numpy as np
    a = np.asarray(unities)
    if a.ndim == 1:          # (n_theta,) -> (1, n_theta)
        return a.reshape(1, a.shape[0])
    if a.ndim != 2:
        raise ValueError(f"Expected (n_group, n_theta) or (n_theta,), got shape {a.shape}")
    return a

class ConcreteAnchors:
    def __init__(self,
                 geo_props: GeoProps,
                 concrete_props: ConcreteProps,
                 anchor_props: Optional[SupportedConcreteAnchors]=None):
        self.geo_props = geo_props
        self.concrete_props = concrete_props
        self.anchor_props = anchor_props
        self.factor_method = anchor_pro.ap_types.FactorMethod.lrfd_omega

        # Compute Tension Groups and create list of TensionGroup objects
        # Compute Shear Groups

    def set_anchor_props(self,anchor_props:SupportedConcreteAnchors):
        self.anchor_props = anchor_props

        tension_groups = get_tension_groups(
            1.5*self.anchor_props.hef_default,
            self.geo_props.xy_anchors,
            self.geo_props.edge_ordinates)
        shear_groups = get_shear_groups(
            self.geo_props.xy_anchors,
            self.geo_props.edge_ordinates
        )

        object.__setattr__(self.geo_props,'tension_groups', tension_groups)
        object.__setattr__(self.geo_props, 'shear_groups', shear_groups)

    def check_anchor_spacing(self):
        # Check Slab Thickness
        thickness_ok = self.concrete_props.t_slab >= self.anchor_props.hmin

        # Check spacing and edge distance
        ok = True
        ok = ok and (self.geo_props.c_min >= self.anchor_props.c1)
        ok = ok and (self.geo_props.s_min >= self.anchor_props.s2)

        if (self.anchor_props.c2, self.anchor_props.s2) != (self.anchor_props.c1, self.anchor_props.s1):
            if (self.geo_props.c_min == self.anchor_props.c1) and (self.geo_props.s_min < self.anchor_props.s1):
                ok = False
            elif self.geo_props.c_min != self.anchor_props.c1:
                m_min = (self.anchor_props.s2 - self.anchor_props.s1) / (self.anchor_props.c2 - self.anchor_props.c1)
                m_existing = (self.geo_props.s_min - self.anchor_props.s1) / (self.geo_props.c_min - self.anchor_props.c1)
                if m_existing < m_min:
                    '''If the slope from the (c1,s1) point to the (ca,smin) point is less than
                    the slop to the (c2,s2) point, the spacing requirements are not met.'''
                    ok = False

        return SpacingRequirements(
            slab_thickness_ok=thickness_ok,
            edge_and_spacing_ok=ok,
            ok = (thickness_ok and ok)
        )


    def check_a_anchor_tension(self, tg: TensionGroup, forces: NDArray)->SteelTensionCalc:
        """ ACI 17.6.1"""
        Nsa = self.anchor_props.Nsa
        if not Nsa:
            raise Exception('No tabulated value for Nsa. Need to implement calculations')

        demand = np.max(forces[tg.anchor_indices,2,:],axis=0)
        phi = self.anchor_props.phi.saN
        phi_seismic = 1.0
        unities = demand / (phi*phi_seismic*Nsa)

        return SteelTensionCalc(demand=demand,
                                phi=phi,
                                phi_seismic=phi_seismic,
                                Nsa=Nsa,
                                unities=reshape_unities_array(unities))


    def check_b_tension_breakout(self, tg: TensionGroup, forces: NDArray)->TensionBreakoutCalc:  # 17.6.2
        """ACI 17.6.2"""

        ca_array = np.array([tg.cax_neg, tg.cax_pos, tg.cay_neg, tg.cay_pos])
        xy_group = self.geo_props.xy_anchors[tg.anchor_indices,:]
        conc = self.concrete_props
        anch = self.anchor_props

        if sum(ca_array < (1.5 * anch.hef_default)) >= 3:  # 17.6.2.1.2 Edge distance < 1.5hef on 3 or more sides
            ca_max = ca_array[ca_array < 1.5 * anch.hef_default].max()
            s_max = 0
            for x in [xy_group[:, 0].min(), xy_group[:, 0].max()]:
                s_max = max([s_max, tg.sy[x].max()])
            for y in [xy_group[:, 1].min(), xy_group[:, 1].max()]:
                s_max = max([s_max, tg.sx[y].max()])

            hef_limit = max(ca_max / 1.5, s_max / 3)
            hef = hef_limit
        else:
            hef_limit = hef = anch.hef_default

        # Breakout Cone Dimensions 17.6.2.1
        bxN = min([tg.cax_neg, 1.5 * hef]) + \
                   (xy_group[:, 0].max() - xy_group[:, 0].min()) + \
                   min([tg.cax_pos, 1.5 * hef])

        byN = min([tg.cay_neg, 1.5 * hef]) + \
                   (xy_group[:, 1].max() - xy_group[:, 1].min()) + \
                   min([tg.cay_pos, 1.5 * hef])

        Anc = float(bxN * byN)
        Anco = float(9 * hef ** 2)

        # Basic Breakout Strength 17.6.2.2
        if ((tg.n_anchor == 1)
            and (self.anchor_props.info.anchor_type in (AnchorTypes.headed_stud, AnchorTypes.headed_bolt))
            and (11.0 <= hef <= 25.0)):  # ACI 17.6.2.2.3
            Nb = 16 * conc.lw_factor_a * (conc.fc ** 0.5) * (hef ** (5 / 3))
        else:
            Nb = self.anchor_props.kc * conc.lw_factor_a * conc.fc ** 0.5 * hef ** 1.5

        # Breakout Eccentricity Factor 17.6.2.3
        t_anchor = np.maximum(forces[tg.anchor_indices, 2, :], 0.0)  #(n_g,n_theta)
        ex, ey = group_eccentricity(xy_group, t_anchor)

        denom = 1.5 * hef
        psi_ecNx = np.minimum(1.0, 1.0 / (1.0 + ex / denom))
        psi_ecNy = np.minimum(1.0, 1.0 / (1.0 + ey / denom))
        psi_ecN = psi_ecNx * psi_ecNy

        # Breakout Edge Factor 17.6.2.4
        psi_edN = min(1.0, 0.7 + 0.3 * tg.ca_min / (1.5 * hef))

        # Breakout cracking factor 17.6.2.5
        psi_cN = 1.0  # In general this factor will be 1.0, when used with mfr-provided kc values

        # Breakout splitting factor 17.6.2.6
        psi_cpN = min(1.0, max(tg.ca_min / self.anchor_props.cac,
                                 1.5 * hef / self.anchor_props.cac))

        # Breakout Strength
        Ncb = (Anc / Anco) * psi_ecN * psi_edN * psi_cN * psi_cpN * Nb

        # Unity Ratio
        demand = t_anchor.sum(axis=0)  #(n_theta,)
        phi = self.anchor_props.phi.cN
        phi_seismic = self.anchor_props.phi.seismic
        unities = demand/(phi*phi_seismic*Ncb)

        return TensionBreakoutCalc(
            hef_default=anch.hef_default,
            hef_limit=hef_limit,
                            hef=hef,
                            bxN=bxN,
                            byN=byN,
                            Anc=Anc,
                            Anco=Anco,
                            Nb=Nb,
                            Ncb=Ncb,
                            ex=ex,
                            ey=ey,
                            psi_ecNx=psi_ecNx,
                            psi_ecNy=psi_ecNy,
                            psi_ecN = psi_ecN,
                            psi_edN= psi_edN,
                            psi_cN=psi_cN,
                            psi_cpN=psi_cpN,
                            demand=demand,
                            phi=phi,
                            phi_seismic=phi_seismic,
                            unities=reshape_unities_array(unities)
                         )

    def check_c_tension_pullout(self, tg: TensionGroup, forces: NDArray) -> TensionPulloutCalc:
        """ ACI 17.6.3"""
        # todo [PULLOUT]: need to impliment pullout amplification factor (fc/fc_min)**b (Compare ESR and Bond)
        Np = self.anchor_props.Np
        if not Np:
            raise Exception(f'No tabulated value for Np for {self.anchor_props.info.anchor_id}. Need to implement calculations')

        demand = np.max(forces[:, 2, :], axis=0)
        phi = self.anchor_props.phi.cN
        phi_seismic = self.anchor_props.phi.seismic
        unities = demand / (phi * Np)

        return TensionPulloutCalc(demand=demand,
                                phi=phi,
                                phi_seismic=phi_seismic,
                                Np=Np,
                                unities=reshape_unities_array(unities))

    def check_d_side_face_blowout(self, tg: TensionGroup, forces: NDArray) -> BlowoutCalc:
        """ACI 17.6.4"""
        #todo [HEADED STUDS]: implement
        conc = self.concrete_props
        anch = self.anchor_props

        # OLD CODE
        # if min([tg.cax_neg, tg.cax_pos]) < min([tg.cay_neg, tg.cay_pos]):
        #     ca1 = min([tg.cax_neg, tg.cax_pos])
        #     ca2 = min([tg.cay_neg, tg.cay_pos])
        #     s1 = tg.sx
        # else:
        #     ca2 = min([tg.cax_neg, tg.cax_pos])
        #     ca1 = min([tg.cay_neg, tg.cay_pos])
        #     s1 = tg.sy
        #
        # factor = max([1.0, min([3.0, ca2 / ca1])])  # 17.6.4.1.1
        # Nsb = factor * 160 * ca1 * anch.abrg ** 0.5 * conc.lw_factor_a * conc.fc ** 0.5
        # Nsbg = (1 + s1 / (6 * ca1)) * Nsb

        # Additional new code
        # demand = 1 # Demand from only anchors where ca < 2.5 hef
        # phi = anch.phi.cN
        # phi_seismic = anch.phi.seismic
        # unity = demand/Nsbg


        raise NotImplementedError

    def check_e_bond_strength(self, tg: TensionGroup, forces: NDArray) -> BondStrengthCalc:
        # todo: [ADHESIVE] add adhesive anchor check
        # Note, will need to adjust lw_factor_a in this calc by using local
        # lw_factor_a = 0.6*self.lw_factor, not self.lw_factor_a
        # See table 17.2.4.1. Lw  factor is specific to bond failure mode.

        # Determine Influence Areas (See ACI Fig R17.6.5.1)

        # self.cna = 10 * self.da * (self.tau_anchor / 1100)  # ACI 17.6.5.1.2b
        # self.Anao = (2 * self.cna) ** 2  # ACI 17.6.5.1.2a
        #
        # self.bxNa = min([self.cax_neg, self.cna]) + \
        #             (self.xy_group[:, 0].max() - self.xy_group[:, 0].min()) + \
        #             min([self.cax_pos, self.cna])
        #
        # self.byNa = min([self.cay_neg, self.cna]) + \
        #             (self.xy_group[:, 1].max() - self.xy_group[:, 1].min()) + \
        #             min([self.cay_pos, self.cna])
        #
        # self.Ana = self.bxNa * self.byNa
        #
        # self.Nba = self.lw_factor_bond_failure * self.tau_anchor * np.pi * self.da * self.hef
        #
        # self.Na = (self.Ana / self.Anao) * self.psi_ecNa * self.psi_edNa * self.psi_cpNa * self.Nba

        # New Code
        # # Breakout Eccentricity Factor 17.6.5.3.1
        # t_anchor = np.maximum(forces[tg.anchor_indices, 0, :], 0.0)  # (n_g,n_theta)
        # ex, ey = group_eccentricity(xy_group, t_anchor)
        #
        # denom = cNa
        # psi_ecNx = np.minimum(1.0, 1.0 / (1.0 + ex / denom))
        # psi_ecNy = np.minimum(1.0, 1.0 / (1.0 + ey / denom))
        # psi_ecN = psi_ecNx * psi_ecNy

        raise NotImplementedError

    def check_f_anchor_shear(self, sg: TensionGroup, forces: NDArray)->SteelShearCalc:
        """ Steel shear is checked by tension group
        because in the case of infinite edge distances, no shear group will exist,
        but steel shear failure should still be checked"""

        Vsa = self.anchor_props.Vsa
        if not Vsa:
            raise Exception('No tabulated value for Vsa. Need to implement calculations')
        # todo: [ADHESIVE] Inclusion of 17.7.1.2.1 anchor shear (for non-tabulated Vsa)

        V = _shear_resultant(forces[sg.anchor_indices,:,:])
        phi = self.anchor_props.phi.saV
        phi_seismic = 1.0 # No seismic reduction per 17.10.6. But ACI 355.3R gives example with 0.75 applied  #todo [WC] verify if a different phi_seismic is required for shear
        unities = V/(phi*phi_seismic*Vsa)

        return SteelShearCalc(demand=V,
                              phi=phi,
                              phi_seismic=phi_seismic,
                              Vsa=Vsa,
                              unities=reshape_unities_array(unities))

    def check_g_shear_breakout(self, sg: ShearGroup, forces: NDArray)->ShearBreakoutCalc:

        conc = self.concrete_props
        anch = self.anchor_props
        xy_group = self.geo_props.xy_anchors[sg.anchor_indices,:]

        ha = conc.t_slab  # Slab thickness

        # Adjust ca1 per 17.7.2.1.2
        # no need to consider ca2 < 1.5 ca1 here. This is already incorporated into how the shear groups are defined
        # if all(np.array([ha, sg.ca2p, sg.ca2n]) < 1.5 * sg.ca1):
            # ca1_eff = max(ha / 1.5, sg.ca2p / 1.5, sg.ca2n / 1.5, sg.s_max / 3)
        if ha < (1.5*sg.ca1):
            ca1_eff = max(ha/1.5, sg.s_max/3)
        else:
            ca1_eff = sg.ca1

        # Calculate Avco per 17.7.2.1.3
        Avco = 4.5 * ca1_eff ** 2

        # Calculate Avc breakout cone area
        b = sg.ca2p_ordinate - sg.ca2n_ordinate
        ha_eff = min(ha, 1.5 * ca1_eff)
        Avc = b * ha_eff

        # Basic shear breakout strength
        Vb = min((7 * (anch.le / anch.da) ** 0.2 * anch.da ** 0.5), 9) * \
                                    conc.lw_factor_a * (conc.fc ** 0.5) * (ca1_eff ** 1.5)

        # Breakout Eccentricity Factor 17.7.2.3
        if (sg.direction == ShearDirLabel.XP) or (sg.direction == ShearDirLabel.XN):
            _, ecc = group_eccentricity(xy_group,forces[sg.anchor_indices,0,:])
        elif (sg.direction == ShearDirLabel.YP) or (sg.direction == ShearDirLabel.YN):
            ecc, _ = group_eccentricity(xy_group, forces[sg.anchor_indices, 1, :])
        else:
            raise Exception("Invalid direction specified for Shear Group.")
        psi_ecV = 1 / (1 + ecc / (1.5 * sg.ca1))


        # Breakout Edge Effect Factor 17.7.2.4
        numerator = min(sg.ca2p,sg.ca2n)
        denominator = 1.5 * sg.ca1
        if np.isinf(numerator) and np.isinf(denominator):  # Handle the division, considering possible infinities
            ratio = np.nan  # Both are infinite, so the ratio is undefined (nan)
        elif np.isinf(denominator) and not np.isinf(numerator):
            ratio = 0.0  # Finite numerator over infinite denominator -> ratio is 0
        else:
            ratio = numerator / denominator
        psi_edV = min(1.0, 0.7 + 0.3 * ratio)

        # Breakout Cracking Factor 17.7.2.5
        psi_cV = 1.0
        # Todo: [Calc Refinement] This factor taken as 1.0, conservatively.
        #  possible to justify higher value with additional user inputs.

        # Breakout Thickness Factor 17.7.2.6
        psi_hV = max((1.5 * sg.ca1 / ha) ** 0.5, 1)

        # Breakout Capacity
        Vcb = (Avc / Avco) * psi_ecV * psi_edV * psi_cV * psi_hV * Vb
        phi = anch.phi.cV
        phi_seismic = anch.phi.seismic

        dir_vecs = {ShearDirLabel.XP: [1,0],
                    ShearDirLabel.XN: [-1,0],
                    ShearDirLabel.YP: [0,1],
                    ShearDirLabel.YN: [0,-1]}
        dir_vec = dir_vecs[sg.direction]

        demand = np.sum(forces[sg.anchor_indices,0,:]*dir_vec[0] + forces[sg.anchor_indices,1,:]*dir_vec[1],axis=0)


        capacity =(phi*phi_seismic*Vcb)  # (1, n_theta)
        unities = np.repeat(np.clip(
            demand[None,:] / capacity[None,:],0,None),
            repeats=sg.n_anchor,axis=0)

        return ShearBreakoutCalc(
            ca1_eff = ca1_eff,
            # ca2p_eff = ca2p_eff,
            # ca2n_eff = ca2n_eff,
            Avco = Avco,
            b = b,
            ha_eff = ha_eff,
            Avc = Avc,
            ecc = ecc,
            psi_ecV = psi_ecV,
            psi_edV = psi_edV,
            psi_cV = psi_cV,
            psi_hV = psi_hV,
            Vb = Vb,
            Vcb = Vcb,
            phi = phi,
            phi_seismic = phi_seismic,
            demand = demand,
            unities=reshape_unities_array(unities)
            )


    def check_h_shear_pryout(self, tg:TensionGroup,forces: NDArray,Np:float)->PryoutCalc:
        kcp = 1.0 if self.anchor_props.hef_default < 2.5 else 2.0  #todo [PROVISIONS QA]: verify if hef needs to be reduced as it is for tension breakout
        Vcp = kcp * Np
        demand = _shear_resultant(forces[tg.anchor_indices,:,:])
        phi = self.anchor_props.phi.cV
        phi_seismic = self.anchor_props.phi.seismic
        unities = demand / (phi*phi_seismic*Vcp)

        return PryoutCalc(demand=demand,
                          phi=phi,
                          phi_seismic=phi_seismic,
                          kcp = kcp,
                          Vcp=Vcp,
                          unities=reshape_unities_array(unities))

    def evaluate(self, forces):
        anch_type = self.anchor_props.info.anchor_type
        # Loop over all tension and shear groups
#           extract demand as slice of anchor forces
#           calculate group capacity
            # calculate group unity
            # return value broadcast to all group anchors
            # tension_unity_by_group -> tension_unity_by_anchor
            # Need: tension unity by anchor
            # Want: unity_by_anchor


        # Calc Results
        steel_tension_calcs = []
        tension_breakout_calcs = []
        anchor_pullout_calcs = []
        side_face_blowout_calcs = []
        bond_strength_calcs = []
        steel_shear_calcs = []
        shear_breakout_calcs = []
        shear_pryout_calcs = []

        # Unity Results
        n_anchor, n_forces, n_theta = forces.shape
        tension_unity = np.zeros((n_anchor, n_theta), dtype=float)
        pryout_unity = np.zeros((n_anchor, n_theta), dtype=float)
        steel_shear_unity = np.zeros((n_anchor,n_theta), dtype=float)
        shear_unity = np.full((n_anchor, n_theta), -np.inf, dtype=float)

        governing_sg_by_anchor = np.full((n_anchor,n_theta), -1, dtype=int)
        tg_by_anchor = np.empty(n_anchor, dtype = int)

        # Calculate Tension Limit States
        for tg_idx, tg in enumerate(self.geo_props.tension_groups):
            # Populate group mapping array
            tg_by_anchor[tg.anchor_indices] = tg_idx

            # Anchor Steel Tension
            steel_tension_calc = self.check_a_anchor_tension(tg, forces)

            # Tension Breakout
            tension_breakout_calc = self.check_b_tension_breakout(tg, forces)

            # Anchor Pullout
            pullout_calc = None
            if self.anchor_props.Np is not None and not np.isnan(self.anchor_props.Np):
                pullout_calc = self.check_c_tension_pullout(tg, forces)

            # Sideface Blowout
            blowout_calc = None
            if anch_type == AnchorTypes.headed_bolt or anch_type == AnchorTypes.headed_stud:
                blowout_calc = self.check_d_side_face_blowout(tg,forces)

            # Adhesive Bond
            if anch_type != AnchorTypes.adhesive:
                bond_calc = None
                Ncp = tension_breakout_calc.Ncb
            else:
                bond_calc = self.check_e_bond_strength(tg,forces)
                Ncp = min(tension_breakout_calc.Ncb, bond_calc.Nag)

            # Shear Pryout (Handled here, because it is derived from Ncb or Na)
            pryout_calc = self.check_h_shear_pryout(tg, forces, Ncp)

            # Steel Tension
            steel_shear_calc = self.check_f_anchor_shear(tg, forces)

            # Append Results
            steel_tension_calcs.append(steel_tension_calc)
            tension_breakout_calcs.append(tension_breakout_calc)
            anchor_pullout_calcs.append(pullout_calc)
            side_face_blowout_calcs.append(blowout_calc)
            bond_strength_calcs.append(bond_calc)
            shear_pryout_calcs.append(pryout_calc)
            steel_shear_calcs.append(steel_shear_calc)

            # Governing Tension Unity
            unities = [
                lim.unities for lim in (
                    tension_breakout_calc,
                    steel_tension_calc,
                    pullout_calc,
                    blowout_calc,
                    bond_calc) if lim is not None]

            tension_unity[tg.anchor_indices,:] = np.maximum.reduce(unities)
            pryout_unity[tg.anchor_indices,:] = pryout_calc.unities
            steel_shear_unity[tg.anchor_indices,:] = steel_shear_calc.unities

        # Calculate Shear Limit States
        for sg_idx, sg in enumerate(self.geo_props.shear_groups):
            # Limit State Calcs
            shear_breakout = self.check_g_shear_breakout(sg, forces)

            # Append Results
            shear_breakout_calcs.append(shear_breakout)

            # Group Unities
            sg_unity = shear_breakout.unities  #(n_group, n_theta)

            current_unities_at_sg_anchors = shear_unity[sg.anchor_indices,:]  #(n_group, n_theta)
            mask = sg_unity > current_unities_at_sg_anchors  #(n_group, n_theta)
            current_unities_at_sg_anchors[mask] = sg_unity[mask]

            shear_unity[sg.anchor_indices,:] = current_unities_at_sg_anchors
            for idx, maskrow in zip(sg.anchor_indices, mask):
                governing_sg_by_anchor[idx,maskrow] = sg_idx

        # Consider Shear Pryout
        shear_unity = np.maximum(shear_unity,pryout_unity, steel_shear_unity)

        # Compute Tension-Shear Interaction
        unities = tension_unity ** (5 / 3) + shear_unity ** (5 / 3)

        # Extract Governing Unity and Limit States
        max_anchor, max_theta = np.unravel_index(np.argmax(unities,axis=None),unities.shape)
        unity = unities[max_anchor,max_theta]
        governing_tension_group = tg_by_anchor[max_anchor]
        governing_shear_group = governing_sg_by_anchor[max_anchor,max_theta]

        return ConcreteAnchorResults(
            tension_groups=self.geo_props.tension_groups.copy(),
            shear_groups=self.geo_props.shear_groups.copy(),
            spacing_requirements = self.check_anchor_spacing(),
            steel_tension_calcs=steel_tension_calcs,
            tension_breakout_calcs = tension_breakout_calcs,
            anchor_pullout_calcs = anchor_pullout_calcs,
            side_face_blowout_calcs = side_face_blowout_calcs,
            bond_strength_calcs = bond_strength_calcs,
            steel_shear_calcs = steel_shear_calcs,
            shear_breakout_calcs = shear_breakout_calcs,
            shear_pryout_calcs = shear_pryout_calcs,
            tension_unity_by_anchor = tension_unity,
            shear_unity_by_anchor = shear_unity,
            unity_by_anchor = unities,
            unity = float(unity),
            ok = unity<1.0,
            governing_tension = forces[max_anchor,0,max_theta],
            governing_shear = np.linalg.norm(forces[max_anchor,1:3,max_theta]),
            governing_tension_group = int(governing_tension_group),
            governing_shear_group = int(governing_shear_group),
            governing_anchor_idx = int(max_anchor),
            governing_theta_idx = int(max_theta),
            forces = forces,
            K = self.anchor_props.K,
            cost_rank = self.anchor_props.info.cost_rank
        )
