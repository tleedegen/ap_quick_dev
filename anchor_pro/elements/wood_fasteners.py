# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 21:35:21 2025

@author: djmiller
"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from anchor_pro.anchor_pattern_mixin import AnchorPatternMixin
from dataclasses import dataclass
from typing import Optional, Literal, Union, Dict, Tuple
from enum import Enum

from anchor_pro.ap_types import FactorMethod
from anchor_pro.utilities import get_anchor_spacing_matrix


class GrainDir(str, Enum):
    X = "X"        # parallel to X
    Y = "Y"        # parallel to Y (90 deg from X)


class MoistureCondition(str, Enum):
    DRY = "Dry"
    WET = "Wet"  # add others if you use them


class WoodClass(str, Enum):
    HARDWOOD = "hardwood"
    SOFTWOOD = "softwood"

class DowelTypes(str, Enum):
    LAG = "Lag Screw"
    SCREW = "Wood Screw"

@dataclass(frozen=True, slots=True)
class MainMemberProps:
    wood_id: Optional[str]
    G: float
    E: float
    moisture: MoistureCondition
    wood_class: Optional[WoodClass]  # hardwood/softwood or None if N/A
    Bm: float  # main member width
    Dm: float  # main member depth
    grain_dir: GrainDir  # 'X' or 'Y'

    @property
    def area(self) -> float:
        return self.Bm * self.Dm

    @property
    def grain_angle(self) -> float:
        # match your earlier dictionary: 'X': 0, 'Y': pi/2
        return 0.0 if self.grain_dir == GrainDir.X else np.pi / 2.0


@dataclass(frozen=True, slots=True)
class SideMemberProps:
    t_steel: float
    Fes: float
    g: float = 0.0  # gap between members
    Es: float = 29000000.0  # psi; override if needed

@dataclass(frozen=True, slots=True)
class WoodFastenerProps:
    fastener_id: Optional[str]
    fastener_type: Literal["Lag Screw", "Wood Screw"]
    D_nom: float  # nominal diameter
    D: float  # shank/root diameter used for calcs
    Fyb: float  # bending yield stress
    length: float
    cost_rank: float


@dataclass(frozen=True, slots=True)
class WoodFastenerResults:
    # Demands
    F: NDArray  # Resultant force
    N: NDArray  # Tension Component
    Vx: NDArray  # Shear X Component
    Vy: NDArray  # Shear Y Component
    V: NDArray  # Combined Shear Component

    # Capacity Calcs
    theta_rel: NDArray[np.floating]  # [n_theta], radians relative to grain
    theta_deg: NDArray[np.floating]  # [n_theta], 0..90
    Fem: NDArray[np.floating]  # [n_theta]
    K_theta: NDArray[np.floating]  # [n_theta]
    K_D: float
    Rd: NDArray[np.floating]  # [n_theta, 6] reduction terms
    yield_modes: Dict[str, NDArray[np.floating]]  # mode -> [n_theta]
    C_M: float
    C_t: float
    C_g: float
    C_delta: float
    C_di: float
    C_tn: float
    C_eg: float
    Kf: float
    phi: float

    Z: NDArray[np.floating]  # [n_theta]
    Z_prime: NDArray[np.floating]  # [n_theta]
    W: float
    W_prime: float
    p: float
    z_alpha_prime: NDArray[np.floating]  # [n_theta, n_anchor]


    # Results
    unities: NDArray
    unity: float
    ok: bool
    governing_theta_idx: int
    governing_fastener_idx: int
    cost_rank: float

class WoodFastener:
    __slots__ = ("member_props", "side_member_props",
                 "fastener_props", "xy_anchors", "max_temp","factor_method")

    def __init__(self,
                 member_props: Optional[MainMemberProps],
                 side_member_props: Optional[SideMemberProps],
                 xy_anchors: NDArray[np.floating],
                 fastener_props: Optional[WoodFastenerProps]=None,
                 max_temp: float = 100.0):
        self.member_props = member_props
        self.side_member_props = side_member_props
        self.fastener_props = fastener_props
        self.xy_anchors = np.asarray(xy_anchors)
        self.max_temp = float(max_temp)
        self.factor_method = FactorMethod.lrfd

    def set_fastener_props(self,fastener_pros:WoodFastenerProps):
        self.fastener_props = fastener_pros

    def _reference_withdrawal(self) -> float:
        fp = self.fastener_props
        mp = self.member_props
        if fp.fastener_type == "Lag Screw":
            # NDS 12.2-1
            return 1800.0 * (mp.G ** 1.5) * (fp.D ** 0.75)
        elif fp.fastener_type == "Wood Screw":
            # NDS 12.2-2
            return 2850.0 * (mp.G ** 2.0) * fp.D
        raise ValueError(f'Unsupported fastener type "{fp.fastener_type}".')

    def _K_D(self) -> float:
        """ K_D from Table 12.3.1B"""
        D = self.fastener_props.D
        if D <= 0.17:
            return 2.2
        if 0.17 < D < 0.25:
            return 10.0 * D + 0.5
        return np.nan  # handled by Rd branch for D >= 0.25

    def _get_Rd(self, K_theta: NDArray, K_D: float) -> NDArray[np.floating]:
        """Return Rd[0..5] for one theta (Table 12.3.1B + note)."""
        fp = self.fastener_props
        D = fp.D
        D_nom = fp.D_nom
        if (D_nom > 0.25) and (D < 0.25):  # footnote 1
            base = K_D * K_theta
            return np.stack([base, base, base, base, base, base])
        if D < 0.25:
            base = np.full_like(K_theta,K_D)
            return np.stack([base, base, base, base, base, base])
        if 0.25 <= D <= 1.0:
            return np.stack([4.0 * K_theta, 4.0 * K_theta,
                             3.6 * K_theta, 3.2 * K_theta,
                             3.2 * K_theta, 3.2 * K_theta])
        # D > 1.0 not permitted
        raise ValueError('Fastener diameter greater than 1" not permitted.')

    def _get_Fem(self, theta_rel: NDArray) -> NDArray:
        """Equivalent wood bearing per NDS 12.3.4 / Table 12.3.3."""
        mp, fp = self.member_props, self.fastener_props
        G = mp.G
        D = fp.D
        if D < 0.25:
            return 16600.0 * (G ** 1.84)
        Fe_parallel = 11200.0 * G
        Fe_perp = 6100.0 * (G ** 1.45) / (D ** 0.5)
        s2, c2 = np.sin(theta_rel) ** 2, np.cos(theta_rel) ** 2
        return (Fe_parallel * Fe_perp) / (Fe_parallel * s2 + Fe_perp * c2)

    def _get_K_theta(self, theta_deg: NDArray) -> NDArray:
        # Table 12.3.1B: linear interpolation: 0°→1.0, 90°→1.25
        return 1.0 + 0.25 * (theta_deg / 90.0)

    def _group_action_factor(self, D: float) -> float:
        """Your derived expression using spacing matrix & stiffness ratio."""
        n = int(self.xy_anchors.shape[0])
        if (D < 0.25) or (n <= 1):
            return 1.0
        spacing = get_anchor_spacing_matrix(self.xy_anchors)
        mp, sp = self.member_props, self.side_member_props
        s = np.where(spacing == 0.0, np.inf, spacing).min()
        if s > mp.Bm:
            return 1.0
            #todo: [WOOD] consider directionality of spacing for group factor
            '''Note, this is nto correct. This returns one if the absolute spacing between all anchors
            is greater than the main member width. but you could have two anchors along the length of the main member.
            Need to refine this to consider directionality of spacing relative to grain direction.'''

        As = mp.Bm * sp.t_steel
        Am = mp.area
        E, Es = mp.E, sp.Es
        gamma = 270000.0 * (D ** 1.5)
        u = 1.0 + gamma * (s / 2.0) * (1.0 / (E * Am) + 1.0 / (Es * As))
        Rea = min(Es * As / (E * Am), E * Am / (Es * As))
        m = u - np.sqrt(max(u * u - 1.0, 0.0))
        # Guard small numerical oddities
        denom = n * ((1 + Rea * (m ** n)) * (1 + m) - 1 + (m ** (2 * n)))
        if denom == 0:
            return 1.0
        Cg = (m * (1 - m ** (2 * n)) / denom) * (1 + Rea) / (1 - m)
        return float(Cg)

    def _yield_Z_for_theta(self, Fem: NDArray, Rd: NDArray[np.floating]) -> Tuple[
        NDArray, Dict[str, NDArray]]:
        """Compute Z(θ) via yield modes (NDS 12.3.1 & TR12 Table 1)."""
        fp, sp = self.fastener_props, self.side_member_props
        D = fp.D
        lm = fp.length - sp.t_steel  # penetration p
        ls = sp.t_steel
        Fes = sp.Fes
        Fyb = fp.Fyb
        g = sp.g

        qs = Fes * D
        qm = Fem * D
        Ms = Fyb * (D ** 3 / 6.0)
        Mm = Fyb * (D ** 3 / 6.0)

        # modes that use a quadratic (A, B, C) / Rd
        modes_ABC = {
            "II": (1 / (4 * qs) + 1 / (4 * qm), ls / 2 + g + lm / 2, qs * (ls ** 2) / 4 - qm * (lm ** 2) / 4,
                   Rd[2,:]),
            "IIIm": (1 / (2 * qs) + 1 / (4 * qm), g + lm / 2, -Ms - qm * (lm ** 2) / 4, Rd[3,:]),
            "IIIs": (1 / (4 * qs) + 1 / (2 * qm), ls / 2 + g, -qs * (ls ** 2) / 4 - Mm, Rd[4,:]),
            "IV": (1 / (2 * qs) + 1 / (2 * qm), g, -Ms - Mm, Rd[5,:]),
        }

        # Single-curvature modes
        yields: Dict[str, NDArray] = {"Im": D * lm * Fem / Rd[0, :], "Is": D * ls * Fes / Rd[1, :]}

        # Quadratic modes
        for k, (A, B, C, Rd_val) in modes_ABC.items():
            disc = np.clip(B * B - 4 * A * C, 0.0,None)
            cap = (-B + np.sqrt(disc)) / (2 * A * Rd_val)
            yields[k] = cap

        Z = np.min(np.stack(list(yields.values())), axis=0)

        return Z, yields

    def evaluate(self,
                 forces: NDArray[np.floating]) -> WoodFastenerResults:
        """
        Parameters
        ----------
        forces: [n_anchor, 3, n_theta] as [T, Vx, Vy]

        Returns
        -------
        WoodFastenerResults
        """

        ''' Inpust and Validation '''
        f = np.asarray(forces, dtype=float)
        if f.ndim != 3 or f.shape[2] < 3:
            raise ValueError("forces must be [n_anchor, 3, n_theta] with [N, Vx, Vy].")
        N = f[:, 0, :]
        Vx = f[:, 1, :]
        Vy = f[:, 2, :]
        n_theta, n_anchor = N.shape

        # Properties
        mp, sp, fp = self.member_props, self.side_member_props, self.fastener_props
        if any(x is None for x in (mp, sp, fp)):
            raise ValueError("Props must be provided before evaluate().")

        # Dowel main member penetration
        p = fp.length - sp.t_steel
        if p <= 0:
            raise ValueError("Fastener penetration (length - side member thickness) must be positive.")

        ''' Withdrawal (W) '''
        W = self._reference_withdrawal()

        ''' ADJUSTMENT FACTORS '''
        # Wet Service Factor (Table 11.3.3)
        C_M = 1.0 if mp.moisture == MoistureCondition.DRY else 0.7
        # Temperature Factor (Table 11.3.4)
        if self.max_temp > 100.0:
            raise ValueError("max_temp > 100°F not supported in current implementation.")
        C_t = 1.0

        # End Grain Factor 12.5.2
        C_eg = 1.0 # Assumed no attachment to member end grain

        # Diaphragm Factor
        C_di = 1.0 # Assumed fasteners are not part of a diaphragm

        # Toe Nailed Factor
        C_tn = 1.0

        # Geometry Factor 12.5.2
        C_delta = 1.0  # assuming away from edges

        # Other Factors
        Kf = 3.32
        phi = 0.65
        time_factor = 1.0

        # Group action factor (geometry + stiffness)
        C_g = self._group_action_factor(fp.D)

        ''' Design Values'''
        # W' (scalar)
        W_prime = W * C_M * C_t * C_eg * Kf * phi * time_factor

        # Z' Parameters
        theta = np.atan2(abs(Vy), abs(Vx))
        theta_rel = theta - mp.grain_angle
        theta_deg = np.clip(np.degrees(np.abs(theta_rel)), 0.0, 90.0)

        K_D = self._K_D()

        # K_theta from table 12.3.1B
        K_theta_vec = self._get_K_theta(theta_deg)

        Rd = self._get_Rd(K_theta_vec, K_D)
        Fem = self._get_Fem(theta_rel)

        Z, yield_modes = self._yield_Z_for_theta(Fem, Rd)
        # Z' per-theta

        Z_prime = Z * C_M * C_t * C_g * C_delta * C_eg * C_di * C_tn * Kf * phi * time_factor

        # --- demands grid ---
        V = np.sqrt(Vx * Vx + Vy * Vy)
        N_pos = np.clip(N, 0.0,None)
        F = np.sqrt(V * V + N_pos * N_pos)

        # orientation weights
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_alpha = np.where(F > 0.0, V / F, 1.0)
            sin_alpha = np.where(F > 0.0, N_pos / F, 0.0)

        # broadcast Z^prime (theta) over anchors
        # Zp = Z_prime.reshape(n_theta, 1)
        Zp = Z_prime
        Wp = float(W_prime)
        denom = (Wp * p) * (cos_alpha ** 2) + Zp * (sin_alpha ** 2)
        # avoid division by zero (when both components 0)
        denom = np.where(denom <= 0.0, np.inf, denom)

        z_alpha_prime = (Wp * p) * Zp / denom

        ''' Unity '''
        with np.errstate(divide="ignore", invalid="ignore"):
            unities = np.where(z_alpha_prime > 0.0, V / z_alpha_prime, 0.0)
        governing_fastener_idx, governing_theta_idx = np.unravel_index(np.argmax(unities, axis=None), unities.shape)
        unity = unities.max()
        ok = unity <= 1

        return WoodFastenerResults(N=N_pos, Vx=Vx, Vy=Vy, V=V, F=F,
                                   Z=Z, Z_prime=Z_prime, W=W, W_prime=W_prime, p=p,
                                   z_alpha_prime=z_alpha_prime,
                                   theta_rel=theta_rel, theta_deg=theta_deg,
                                   Fem=Fem, K_theta=K_theta_vec, K_D=float(K_D),
                                   Rd=Rd, yield_modes=yield_modes,
                                   C_M= C_M,
                                   C_t= C_t,
                                   C_g= C_g,
                                   C_delta= C_delta,
                                   C_di= C_di,
                                   C_tn= C_tn,
                                   C_eg= C_eg,
                                   Kf= Kf,
                                   phi= phi,
                                   unities=unities,
                                   unity = unity,
                                   governing_theta_idx=int(governing_theta_idx),
                                   governing_fastener_idx=int(governing_fastener_idx),
                                   ok=ok,
                                   cost_rank=self.fastener_props.cost_rank)

class WoodFastener_OLD(AnchorPatternMixin):
    EDGE_DIST_REQS = pd.DataFrame({
        'loading_dir':['perpendicular', 'parallel_compression','parallel_tension', 'parallel_tension'],
        'wood_class':[None, None, 'softwood', 'hardwood'],
        'minimum_for_05': [2, 2, 3.5, 2.5],
        'minimum_for_1': [4, 4, 7 ,5]
    })

    GRAIN_ANGLE = {'X': 0,
                   'Y': np.pi/2}

    def __init__(self, xy_anchors):
        # General Properties
        self.xy_anchors = xy_anchors
        self.anchor_forces = None
        self.Tu_max = None  # Tu_max is not used in calculations, but is referenced by function in the report in order to select the governing anchor object.
        self.max_temp = 100
        self.spacing_matrix = None
        
        # Wood Properties (Main Member)
        self.wood_id = None
        self.G = None
        self.E = None
        self.moisture_condition = "Dry"
        self.hardwood_softwood = None  # hardwood or softwood
        self.Bm = None
        self.Dm = None
        self.Am = None
        self.grain_angle = None

        self.theta = None  # Angle w.r.t wood grain
        self.Fe_parallel = None
        self.Fe_perp = None
        self.Fem = None
        self.lm = None

        # "Side Member" Properties (Metal Attachment)
        self.t_steel = None
        self.Fes = None
        self.As = None
        self.g = None  # (Gap between members)
        self.Es = None

        # Fastener Properties
        self.fastener_id = None
        self.fastener_type = None  # "Lag Screw" or "Wood Screw"
        self.D_nom = None
        self.D = None
        self.Fyb = None
        self.length = None

        # Withdrawal Strength Computed Properties
        self.p = None
        self.W = None

        # Lateral Strength Computed Properites
        self.K_theta = None
        self.K_D = None
        self.Rd = None
        self.yield_modes = {}
        self.Z = None

        # Combined Strength Computed Properties
        self.alpha = None
        self.Z_prime = None
        self.W_prime = None
        self.C_M = None
        self.C_t = None
        self.C_g = None
        self.C_delta = None
        self.C_eg = None
        self.C_di = None
        self.C_tn = None
        self.Z_alpha = None

        self.z_alpha_prime = None
        self.V = None
        self.Vy = None
        self.Vx = None
        self.N = None
        self.DCR = None

    def set_member_properties_from_data_table(self, member_data, base_or_wall='base'):
        """ base_or_wall is a key that must be appended to the column parameter
        names to get the correct column names for the equipment data table"""

        # Properties for input tables
        self.G = member_data['G_wood'+'_'+base_or_wall]
        self.E = member_data['E_wood'+'_'+base_or_wall]
        self.moisture_condition = member_data['moisture_condition_wood'+'_'+base_or_wall]
        self.hardwood_softwood = member_data['hardwood_softwood'+'_'+base_or_wall]  # hardwood or softwood
        self.Bm = member_data['Bm'+'_'+base_or_wall]
        self.Dm = member_data['Dm'+'_'+base_or_wall]

        # Computed Properties
        self.grain_angle = WoodFastener_OLD.GRAIN_ANGLE[member_data['grain_direction'+'_'+base_or_wall]]
        self.Am = self.Bm * self.Dm


    def set_steel_props(self, t_steel, Fes, g=0):
        # Passed parameters
        self.t_steel = t_steel
        self.Fes = Fes
        self.g = g

        # Inferred parameters
        self.Es = 29000
        self.As = self.Bm * self.t_steel


    def set_fastener_properties(self, fastener_data):
        for key in vars(self).keys():
            if key in fastener_data.keys():
                setattr(self, key, fastener_data.at[key])

        # Calculate penetration values
        self.p = self.length - self.t_steel

    def reference_lateral_design_value(self):
        """ Reference Technical Report 12 for inclusion of gap offset due to gyp board"""
        
        D = self.D
        lm = self.p
        ls = self.t_steel
        Fem = self.Fem
        Fes = self.Fes
        Fyb = self. Fyb # Get this from fastener data
        g = self.g
        
        Rd = self.Rd
        # Rt = lm/ls
        # Re = Fem/Fes

        # k1 = (Re + 2*Re**2*(1 + Rt + Rt**2) + Rt**2*Re**3)**0.5 - Re*(1+Rt) / (1+Re)
        # k2 = -1 + (2 * (1 + Re) + 2 * Fyb * (1 + 2 * Re) * D ** 2 / (3 * Fem * lm ** 2)) ** 0.5
        # k3 = -1 + (2 * (1 + Re) / Re + 2 * Fyb * (2 + Re) * D ** 2 / (3 * Fem * ls ** 2)) ** 0.5

        # Yield Limit Equations (NDS 12.3.1 and TR12 Table 1)
        qs = Fes*D
        qm = Fem*D
        Ms = Fyb * (D**3/6)
        Mm = Fyb * (D**3/6)
        yield_modes_ABCRd = {'II': (1/(4*qs) + 1/(4*qm), ls/2 + g + lm/2, qs*ls**2/4 - qm*lm**2/4, Rd[2]),
                           'IIIm': (1/(2*qs)+1/(4*qm), g + lm/2, -Ms-qm*lm**2/4, Rd[3]),
                           'IIIs': (1/(4*qs) + 1/(2*qm), ls/2 + g, -qs*ls**2/4-Mm, Rd[4]),
                           'IV': (1/(2*qs) + 1/(2*qm), g, -Ms-Mm, Rd[5])}
        
        self.yield_modes['Im'] = D * lm * Fem / Rd[0]
        self.yield_modes['Is'] = D * ls * Fes / Rd[1]
        for mode, (A, B, C, Rd) in yield_modes_ABCRd.items():
            self.yield_modes[mode] = (-B + (B**2 - 4*A*C)**0.5)/(2*A*Rd)
                            # 'II': k1 *  D * ls * Fes / Rd[2],
                            # 'IIIm': k2 * D * lm * Fem / ( (1 + 2 * Re) * Rd[3]),
                            # 'IIIs': k3 * D * ls * Fem / ( (2 + Re) + Rd[4]),
                            # 'IV': D**2 / Rd[5] * (2 * Fem * Fyb / (3 * (1 + Re)))**0.5}

        self.Z = min([val for key, val in self.yield_modes.items()])



    def reference_withdrawal_design_value(self):
        if self.fastener_type == 'Lag Screw':
            self.W = 1800*self.G**(3/2) * self.D**(3/4)  # NDS 12.2-1
        elif self.fastener_type == 'Wood Screw':
            self.W = 2850 * self.G**2 * self.D  # NDS 12.2-2
        else:
            raise Exception(f'Wood fastener type "{self.fastener_type}" not supported.')

    def get_loading_dir(self, Vx, Vy):
        """ Determines the direction of loading relative to the wood grain"""
        theta_load = np.atan2(abs(Vy),abs(Vx))
        self.theta = theta_load - self.grain_angle # Angle w.r.t wood grain

        '''COMPUTED PROPERITES'''
        theta_degrees = np.degrees(self.theta)

        # Wood Bearing (NDS 12.3.4 and Table 12.3.3)
        if self.D < 0.25:
            self.Fem = 16600 * self.G ** 1.84
        else:
            self.Fe_parallel = 11200 * self.G
            self.Fe_perp = 6100 * self.G ** 1.45 / (self.D ** 0.5)
            self.Fem = self.Fe_parallel * self.Fe_perp / (self.Fe_parallel * np.sin(self.theta) ** 2 +
                                                          self.Fe_perp * np.cos(self.theta) ** 2)

        # K_theta from table 12.3.1B
        self.K_theta = 1 + 0.25 * (theta_degrees / 90)

        # K_D from Table 12.3.1B
        if self.D <= 0.17:
            self.K_D = 2.2
        elif 0.17 < self.D < 0.25:
            self.K_D = 10 * self.D + 0.5
        else:
            self.K_D = np.nan

        # Reduction term Rd from Table 12.3.1B
        if self.D < 0.25:
            self.Rd = [self.K_D] * 6
        elif (self.D_nom > 0.25) and (self.D < 0.25):  # footnote 1
            self.Rd = [self.K_D * self.K_theta] * 6
        elif 0.25 <= self.D <= 1:
            self.Rd = [4 * self.K_theta] * 2 + [3.6 * self.K_theta] + [3.2 * self.K_theta] * 3
        else:
            self.Rd = [np.inf] * 6
            raise Exception("Fastener diameter greater than 1\" not permitted")


    def adjustment_factors(self):

        # Wet Service Factor (Table 11.3.3)
        self.C_M = 1.0 if self.moisture_condition == 'Dry' else 0.7  
        
        # Temperature Factor (Table 11.3.4)
        if self.max_temp <= 100:
            self.C_t = 1.0
        else:
            raise Exception('The specified max temperature is not supported')
        
        # Group Action Factor
        if self.D<0.25 or self.xy_anchors.shape[0] == 1:
            self.C_g = 1.0
        else:            
            # xy_extents = max(self.xy_anchors.max(axis=0)-self.xy_anchors.min(axis=0))
            self.spacing_matrix = self.get_anchor_spacing_matrix(self.xy_anchors)
            As = self.As
            Am = self.Am
            n = len(self.xy_anchors)
            s = np.where(self.spacing_matrix==0,np.inf,self.spacing_matrix).min()
            gamma = 270000 * self.D**1.5
            u = 1 + gamma*(s/2)*(1/(self.E*Am) + 1/(self.Es*As))
            Rea = min(self.Es*As/(self.E*Am), self.E*Am/(self.Es*As))
            m = u - (u**2 - 1)**0.5
            self.C_g = (m * (1 - m**(2*n)) / (n*((1+Rea*m**n)*(1+m)-1+m**(2*n))) ) * (1+Rea)/(1-m)
        
        
        # Geometry Factor 12.5.2
        edge_dist_reqs = {'perpendicular': (2*self.D, 4*self.D),
                          'parallel_compression': (2*self.D, 4*self.D),
                          'parallel_tension_hardwood': (2.5*self.D, 5*self.D),
                          'parallel_tension_softwood': ()}

        if self.D < 0.25:
            self.C_delta = 1.0
        else:
            # Assume fastener is located away from edges
            self.C_delta = 1.0

        # End Grain Factor 12.5.2
        self.C_eg = 1.0  # Assumed no attachment to member end grain
        
        # Diaphragm Factor
        self.C_di = 1.0  # Assumed fasteners are not part of a diaphragm
        
        # Toe Nailed Factor
        self.C_tn = 1.0
        
        self.Kf = 3.32
        self.phi = 0.65
        self.time_factor = 1.0  # Table N3

    def check_fasteners(self):  # todo: Refactor to pass xy_anchors as parameter
        Tu_values = np.linalg.norm(self.anchor_forces,axis=2)
        idx_governing_fastener, idx_governing_theta = np.unravel_index(np.argmax(Tu_values),
                                                  self.anchor_forces[:,:,0].shape)
        self.Tu_max = Tu_values[idx_governing_fastener,idx_governing_theta]

        self.reference_withdrawal_design_value()
        self.adjustment_factors()  #todo: Refactor Pass xy_anchors here (used for group effect)
        self.DCR = 0
        for N, Vx, Vy in self.anchor_forces[idx_governing_fastener,:,:]:
            self.get_loading_dir(Vx, Vy)
            self.reference_lateral_design_value()
            Z_prime = self.Z * self.C_M * self.C_t * self.C_g * self.C_delta * self.C_eg * self.C_di * self.C_tn * self.Kf * self.phi * self.time_factor
            W_prime = self.W * self.C_M * self.C_t * self.C_eg * self.Kf * self.phi * self.time_factor

            shear_demand = (Vx**2 + Vy**2)**0.5
            tension_demand = max(N, 0)
            total_demand = (Vx**2 + Vy**2 + N**2)**0.5
            if np.isclose(total_demand,0):
                cos_alpha = 1
                sin_alpha = 0
            else:
                cos_alpha = shear_demand / total_demand
                sin_alpha = tension_demand / total_demand

            z_alpha_prime = W_prime * self.p * Z_prime / (W_prime*self.p*cos_alpha**2+Z_prime*sin_alpha**2)
            dcr = shear_demand/z_alpha_prime
            if dcr > self.DCR:
                self.N = N
                self.Vx = Vx
                self.Vy = Vy
                self.V = shear_demand
                self.Z_prime = Z_prime
                self.W_prime = W_prime
                self.z_alpha_prime = z_alpha_prime
                self.DCR = dcr
