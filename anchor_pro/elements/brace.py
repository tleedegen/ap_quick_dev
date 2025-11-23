from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from anchor_pro.ap_types import SupportingPlanes

@dataclass(frozen=True,slots=True)
class BraceReleases:
    """Releases at the connection/attachment point and along directions."""
    TENSION: bool = False
    COMPRESSION: bool = False

    @property
    def any(self):
        return any([self.TENSION, self.COMPRESSION])

@dataclass(frozen=True,slots=True)
class GeoProps:
    # Geometry (global coordinates)
    xyz_equipment:NDArray
    xyz_support:NDArray
    supporting_plane: SupportingPlanes
    releases: BraceReleases

    # normal_unit_vector: NDArray  # shape (3,) unit vector (n̂) wall→equipment
    support_normal_vector: Optional[NDArray] = None  # shape (3,), vector normal
    C: Optional[NDArray] = None  # DOF Constraint matrix
    A: Optional[NDArray] = None  # Kinematic matrix
    r_support_to_equip: Optional[NDArray] = None
    length: Optional[float] = None
    K_norm: Optional[NDArray] = None  # Pre-computed stiffness matrix normalized by K_norm
    # Derived (computed once)
    def __post_init__(self):
        # Ensure geometry inputs are arrays, not lists or tuples.
        # The object.__setattr__ syntax is required to set attributes of frozen dataclasses.
        object.__setattr__(self, "xyz_equipment", np.asarray(self.xyz_equipment, dtype=float))
        # object.__setattr__(self, "normal_unit_vector", np.asarray(self.normal_unit_vector, dtype=float))

        x, y, z = self.xyz_equipment
        C = np.array([[1.0, 0.0, 0.0, 0.0, z, -y],
                      [0.0, 1.0, 0.0, -z, 0.0, x],
                      [0.0, 0.0, 1.0, y, -x, 0.0]], dtype=float)
        object.__setattr__(self, "C", C)  # (3×6), maps equipment 6DOF → transl. at equipment point

        dx, dy, dz = self.xyz_equipment - self.xyz_support
        length = float(np.linalg.norm(self.xyz_equipment - self.xyz_support))
        object.__setattr__(self, "length", length)

        A = np.array([dx/length, dy/length, dz/length])
        object.__setattr__(self, "A", A)  # (3×3), global→local [n,p,z]

        K_norm = C.T@A.T@A@C
        object.__setattr__(self, "K_norm", K_norm)

        # convenience
        r = self.xyz_equipment - self.xyz_backing  # vector from wall backing → equipment
        object.__setattr__(self, "r_wall_to_equip", r)



        a = self.xyz_backing[2]
        b = self.L - a
        wall_flexibility = (a ** 2 + b ** 2) / (3 * self.E * self.I * self.L)  # Wall idealized as simple span
        object.__setattr__(self, "wall_flexibility", wall_flexibility)

@dataclass(frozen=True,slots=True)
class BraceProps:
    pass

@dataclass(frozen=True, slots=True)
class WallBracketResults:
    pass

class BraceElement:
    """ A uniaxial truss-like element"""
    def __init__(self,
                 geo_props: GeoProps,
                 brace_props: Optional[BraceProps]=None):
        self.geo_props = geo_props
        self.brace_props = brace_props
        self.factor_method = None

    def set_brace_props(self, brace_props: BraceProps, factor_method):
        self.brace_props = brace_props
        self.factor_method = factor_method

    def get_element_stiffness_matrix(self, u=None):
        """ Returns a 6x6 stiffness matrix for the primary degrees of freedom. """

        # Compute initial stiffness values
        k_strap = self.brace_props.k

        # If releases are defined and u is provided, check for displacement conditions
        if (u is not None) and self.geo_props.releases.any:
            delta = self.geo_props.G @ (self.geo_props.C @ u[:6])
            r = self.geo_props.releases

            if (delta[0] > 0 and r.NP) or (delta[0] < 0 and r.NN): kn = 0.0
            if (delta[1] > 0 and r.PP) or (delta[1] < 0 and r.PN): kp = 0.0
            if (delta[2] > 0 and r.ZP) or (delta[2] < 0 and r.ZN): kz = 0.0

        # Construct local stiffness matrix
        k_br = np.diag([kn, kp, kz])

        # Compute global stiffness matrix without modifying self
        K = self.geo_props.C.T @ self.geo_props.G.T @ k_br @ self.geo_props.G @ self.geo_props.C
        return K  # , k_br