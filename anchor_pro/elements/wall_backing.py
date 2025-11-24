from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from anchor_pro.ap_types import WallPositions, FactorMethod
from anchor_pro.elements.elastic_bolt_group import ElasticBoltGroupProps, ElasticBoltGroupResults, calculate_bolt_group_forces

class BackingType(str, Enum):
    FLAT = 'Wall Backing (Flat)'
    STRUT = 'Wall Backing (Strut)'

@dataclass(frozen=True, slots=True)
class WallBackingProps:
    # Backing hardware properties
    d: float                               # backing depth (e.g., to wall or standoff)
    backing_type: BackingType              # "Flat", "Angle", etc.
    supporting_wall: WallPositions

    pz_brackets: NDArray # (n_brkt, 2) [p,z] of bracket attachment points on the backing
    xyz_brackets: NDArray  # (n_brkt,3)

    # x_bar: float
    # y_bar: float

    # Steel/material props (optional if checked elsewhere)
    fy: Optional[float] = None
    t_steel: Optional[float] = None

    # Derived
    n_brackets: int = 0  # 0 if none/provided later


    def __post_init__(self):
        n_b = self.xyz_brackets.shape[0]
        object.__setattr__(self,'n_brackets',n_b)




class WallBackingElement:
    def __init__(self, props: WallBackingProps,
                 bolt_group_props: ElasticBoltGroupProps,
                 factor_method: FactorMethod=FactorMethod.lrfd):
        self.props = props  # read-only usage downstream
        self.bolt_group_props = bolt_group_props
        self.factor_method = factor_method

    def evaluate(self, bracket_reactions_glob: NDArray) -> ElasticBoltGroupResults:
        """ bracket_reactions_glob: (n_brkt, 6, n_theta) global coordinates"""

        bg = self.bolt_group_props
        n_t = bracket_reactions_glob.shape[-1]
        n_b = bracket_reactions_glob.shape[0]
        n_a = bg.n_anchors

        # 1) Sum bracket reactions -> centroid resultants (GLOBAL)
        # Forces
        F_glob = bracket_reactions_glob[:, 0:3, :]            # (n_b, 3, n_t)
        M_glob = bracket_reactions_glob[:, 3:6, :]            # (n_b, 3, n_t)

        # position vectors from centroid to bracket points (GLOBAL)
        cent_offset_global = bg.global_to_local_transformation.T @ np.array([bg.anchor_centroid[0],bg.anchor_centroid[1],0])
        bg_centroid_global = bg.plate_centroid_XYZ + cent_offset_global
        r = self.props.xyz_brackets - bg_centroid_global  # (n_b, 3)

        # r x F (broadcast over theta)
        # result shape: (n_b, 3, n_t)
        r_cross_F = np.cross(r[:, :, None], F_glob, axis=1)

        # Sum over brackets
        F_sum_glob = F_glob.sum(axis=0)                       # (3, n_t)
        M_sum_glob = (M_glob + r_cross_F).sum(axis=0)         # (3, n_t)

        # 2) Transform GLOBAL -> LOCAL
        R = bg.global_to_local_transformation
        F_sum_loc = R @ F_sum_glob                             # (3, n_t)
        M_sum_loc = R @ M_sum_glob                             # (3, n_t)

        # Arrange into (6, n_t) in your preferred order
        # Example order: [N, Vp, Vz, Mx, My, T]
        # Map local axes: assume local axes are [x=normal, y=p, z=z] or as per your standard.
        N, Vp, Vz = F_sum_loc[0, :], F_sum_loc[1, :], F_sum_loc[2, :]
        Mx, My, T  = M_sum_loc[0, :], M_sum_loc[1, :], M_sum_loc[2, :]
        centroid_forces_loc = np.vstack([N, Vp, Vz, Mx, My, T])  # (6, n_t)

        # 3) Distribute to anchors (LOCAL)
        anchor_forces = calculate_bolt_group_forces(N, Vp, Vz, Mx, My, T,
                                                    bg.n_anchors,
                                                    bg.inert_props_cent,
                                                    bg.inert_props_x,
                                                    bg.inert_props_y)

        return ElasticBoltGroupResults(
            centroid_forces_loc=centroid_forces_loc,
            anchor_forces_loc=anchor_forces
        )