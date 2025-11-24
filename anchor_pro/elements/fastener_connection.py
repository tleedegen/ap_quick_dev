from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from anchor_pro.ap_types import WallPositions
from anchor_pro.elements.elastic_bolt_group import ElasticBoltGroupProps, ElasticBoltGroupResults, calculate_bolt_group_forces
from anchor_pro.ap_types import FactorMethod


class FastenerConnection:
    def __init__(self, bolt_group_props: ElasticBoltGroupProps):
        self.bolt_group_props = bolt_group_props
        self.factor_method = FactorMethod.asd

    def evaluate(self, cxn_reactions_glob: NDArray) -> ElasticBoltGroupResults:
        """
        cxn_reactions_glob: (6,n_theta) reactions at connection node
        """

        bg = self.bolt_group_props
        n_t = cxn_reactions_glob.shape[-1]

        # 1) Calculate global forces about bolt group centroid
        # Global forces about connection node
        F_glob = cxn_reactions_glob[0:3, :]  # (3, n_t)
        M_glob = cxn_reactions_glob[3:6, :]  # (3, n_t)

        # position vectors from connection node to connection centroid (GLOBAL)
        r = -bg.global_to_local_transformation.T @ np.array([bg.anchor_centroid[0],
                                                             bg.anchor_centroid[1],
                                                             0]) # (3,)

        # r x F (broadcast over theta)
        r_cross_F = np.cross(r[:,None], F_glob, axis=0) # (3,n_t)
        M_sum_glob = (M_glob + r_cross_F)  # (3, n_t)

        # 2) Convert global connection forces to local connection forces
        R = bg.global_to_local_transformation
        F_sum_loc = R @ F_glob  # (3, n_t)
        M_sum_loc = R @ M_sum_glob  # (3, n_t)

        # Arrange into (6, n_t) in your preferred order
        # Example order: [N, Vp, Vz, Mx, My, T]
        # Map local axes: assume local axes are [x=normal, y=p, z=z] or as per your standard.
        N, Vp, Vz = F_sum_loc[2, :], F_sum_loc[0, :], F_sum_loc[1, :]
        Mx, My, T = M_sum_loc[0, :], M_sum_loc[1, :], M_sum_loc[2, :]
        centroid_forces_loc = np.vstack([N, Vp, Vz, Mx, My, T])  # (6, n_t)

        anchor_forces = calculate_bolt_group_forces(N=N, Vx=Vp, Vy=Vz, Mx=Mx, My=My, T=T,
                                                    n_anchors=bg.n_anchors,
                                                    inert_c=bg.inert_props_cent,
                                                    inert_x=bg.inert_props_x,
                                                    inert_y=bg.inert_props_y)

        return ElasticBoltGroupResults(
            centroid_forces_loc=centroid_forces_loc,
            anchor_forces_loc=anchor_forces
        )