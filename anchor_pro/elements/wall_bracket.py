from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from anchor_pro.ap_types import WallPositions, FactorMethod

ReleaseFlags = Tuple[bool, bool, bool, bool, bool, bool]
# Meaning: (pos_n, neg_n, pos_p, neg_p, pos_z, neg_z)

@dataclass(frozen=True,slots=True)
class BracketReleases:
    """Releases at the connection/attachment point and along directions."""
    NP: bool = False
    NN: bool = False
    PP: bool = False
    PN: bool = False
    ZP: bool = False
    ZN: bool = False

    @property
    def any(self):
        return any([self.NP, self.NN, self.PP, self.PN, self.ZP, self.ZN])

@dataclass(frozen=True, slots=True)
class WallOffsets:
    XP: Optional[float]
    XN: Optional[float]
    YP: Optional[float]
    YN: Optional[float]

@dataclass(frozen=True, slots=True)
class GeometryProps:
    # Geometry (global coords)
    xyz_equipment: NDArray  # shape (3,)
    xyz_wall: NDArray  # shape (3,)
    xyz_backing: NDArray # (3,)

    supporting_wall: WallPositions
    normal_unit_vector: NDArray  # shape (3,) unit vector (n̂) pointing equipment→wall or wall→equipment; be consistent.

    # Wall Properties
    E: float  # Wall modulus
    I: float  # Wall moment of inertia (for effective strip)
    L: float  # wall height

    releases: BracketReleases

    connection_normal_vector: Optional[NDArray] = None  # shape (3,), vector normal
    C: Optional[NDArray] = None
    G: Optional[NDArray] = None
    r_wall_to_equip: Optional[NDArray] = None
    length: Optional[float] = None
    wall_flexibility: Optional[float] = None

    # Derived (computed once)
    def __post_init__(self):
        # Ensure geometry inputs are arrays, not lists or tuples.
        # The object.__setattr__ syntax is required to set attributes of frozen dataclasses.
        object.__setattr__(self, "xyz_equipment", np.asarray(self.xyz_equipment, dtype=float))
        object.__setattr__(self, "xyz_wall", np.asarray(self.xyz_wall, dtype=float))
        object.__setattr__(self, "normal_unit_vector", np.asarray(self.normal_unit_vector, dtype=float))

        x, y, z = self.xyz_equipment
        C = np.array([[1.0, 0.0, 0.0, 0.0, z, -y],
                      [0.0, 1.0, 0.0, -z, 0.0, x],
                      [0.0, 0.0, 1.0, y, -x, 0.0]], dtype=float)
        object.__setattr__(self, "C", C)  # (3×6), maps equipment 6DOF → transl. at equipment point

        nx, ny, nz = self.normal_unit_vector
        G = np.array([[nx, ny, 0.0],
                      [-ny, nx, 0.0],
                      [0.0, 0.0, 1.0]], dtype=float)
        object.__setattr__(self, "G", G)  # (3×3), global→local [n,p,z]

        # convenience
        r = self.xyz_equipment - self.xyz_backing  # vector from wall backing → equipment
        object.__setattr__(self, "r_wall_to_equip", r)

        length = float(np.linalg.norm(self.xyz_equipment - self.xyz_wall))
        object.__setattr__(self, "length", length)

        a = self.xyz_backing[2]
        b = self.L - a
        wall_flexibility = (a ** 2 + b ** 2) / (3 * self.E * self.I * self.L)  # Wall idealized as simple span
        object.__setattr__(self, "wall_flexibility", wall_flexibility)

@dataclass(frozen=True, slots=True)
class BracketProps:
    bracket_id: str
    # Flexibility / stiffness
    brace_flexibility: float          # f_brace
    kn: float
    kp: float                         # horizontal stiffness (parallel to wall, local p)
    kz: float                         # vertical stiffness (local z)

    # Capacities
    capacity_to_equipment: Optional[float] = None
    bracket_capacity: Optional[float] = None
    capacity_to_backing: Optional[float] = None
    shear_capacity: Optional[float] = None
    capacity_method: FactorMethod = FactorMethod.asd

@dataclass(frozen=True, slots=True)
class WallBracketResults:
    # Local forces (per load case)
    fn: np.ndarray  # tension (+) / compression (−), shape (n_theta,)
    fp: np.ndarray  # in‑plane shear (local p), shape (n_theta,)
    fz: np.ndarray  # vertical shear (local z), shape (n_theta,)

    # Stiffness (cached for solution cache selection, see project_controller.run)
    kn: float

    # Global reactions: at backing and equipment, 6 dof each (fx,fy,fz,mx,my,mz)
    reactions_backing: np.ndarray  # shape (6, n_theta)
    reactions_equipment: np.ndarray  # shape (6, n_theta)

    # Demand/capacity
    tension_unity: Optional[np.ndarray] = None        # shape (n_theta,) based on fn in tension
    governing_capacity: Optional[float] = None  # scalar governing capacity used for checks
    unity: Optional[float] = None

    governing_theta_idx: Optional[int] = None

    # Intermediates (optional but useful for debugging)
    # delta_local: np.ndarray          # local relative displacements [n,p,z], shape (n_theta,3)
    # active_axes_mask: np.ndarray     # int mask (n_theta,3) where 1 = engaged, 0 = released

    def __post_init__(self):
        if self.tension_unity is not None:
            unity = self.tension_unity.max()
            max_theta  = np.argmax(self.tension_unity)
            object.__setattr__(self, "unity", unity)
            object.__setattr__(self, "governing_theta_idx", max_theta)
        else:
            object.__setattr__(self, "unity", 0)
            object.__setattr__(self, "governing_theta_idx", 0)


class WallBracketElement:
    def __init__(self, geo_props: GeometryProps,
                 bracket_props:Optional[BracketProps]=None,
                 factor_method: FactorMethod=FactorMethod.lrfd):
        self.geo_props = geo_props
        self.bracket_props = bracket_props
        self.factor_method = factor_method

    def set_bracket_props(self, bracket_props:BracketProps):
        self.bracket_props = bracket_props

    def get_element_stiffness_matrix(self, u=None):
        """ Returns a 6x6 stiffness matrix for the primary degrees of freedom without modifying self. """

        # Compute initial stiffness values
        kn = self.bracket_props.kn
        kp = self.bracket_props.kp
        kz = self.bracket_props.kz

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
        return K  #, k_br

    def evaluate(self,
            u_global: NDArray,  # (n_dof, n_theta)
            method: FactorMethod) -> WallBracketResults:

        """
        Vectorized evaluation. Accepts multiple global DOF solutions (deformation-based),
        computes bracket internal forces (via local stiffness reassembly),
        and returns reactions and unity checks.
        """

        u_global = np.asarray(u_global, dtype=float)
        n_theta = u_global.shape[1]

        # Extract the 6 DOF for this element at the equipment connection
        Ue = u_global[0:6, :]  # (n_theta, 6)

        # Local relative displacement at bracket line: delta = G @ (C @ u_equip)
        delta_XYZ = self.geo_props.C @ Ue  # (6,3) @ (6, n_theta) = (3, n_theta)
        delta_npz = self.geo_props.G @ delta_XYZ   # (3,3) @ (3, n_theta) @= (3, n_theta) columns = [delta_n, delta_p, delta_z]

        # Base stiffness components
        kn0 = self.bracket_props.kn
        kp0 = self.bracket_props.kp
        kz0 = self.bracket_props.kz

        # Apply releases (if any) by turning off stiffness based on sign of delta

        r = self.geo_props.releases

        dn = delta_npz[0, :]
        dp = delta_npz[1, :]
        dz = delta_npz[2, :]

        # For each axis, if sign triggers a release, set engaged=0 for that case

        kn = np.where((dn > 0) & r.NP, 0, np.where((dn < 0) & r.NN, 0, kn0))
        kp = np.where((dp > 0) & r.PP, 0, np.where((dp < 0) & r.PN, 0, kp0))
        kz = np.where((dz > 0) & r.ZP, 0, np.where((dz < 0) & r.ZN, 0, kz0))

        # Local forces: f_local = k_local * delta_local (axiswise)
        fn = kn * delta_npz[0, :]
        fp = kp * delta_npz[1, :]
        fz = kz * delta_npz[2, :]

        # Back to global translational force at backing: Fg_backing = G^T @ f_local
        npz_forces = np.stack([fn, fp, fz], axis=0)  # (3, n_theta)
        backing_forces = np.dot(self.geo_props.G.T, npz_forces)
        equipment_forces = -1 * backing_forces
        moment_reactions = np.cross(self.geo_props.r_wall_to_equip, backing_forces.T).T

        reactions_equipment = np.concatenate((equipment_forces, moment_reactions),axis=0)  # (6,n_theta)
        reactions_backing = np.concatenate((backing_forces, moment_reactions),axis=0)  # (6,n_theta)

        # Capacity checks (tension only on fn)
        capacities = [self.bracket_props.capacity_to_equipment,
                      self.bracket_props.bracket_capacity,
                      self.bracket_props.capacity_to_backing]
        caps_num = [c for c in capacities if isinstance(c, (int, float)) and c is not None]
        governing_capacity = min(caps_num) if caps_num else None

        if governing_capacity and method==self.bracket_props.capacity_method:
            demand = np.clip(fn, 0.0, None)
            tension_unity = demand / governing_capacity
        else:
            tension_unity=None

        return WallBracketResults(fn=fn, fp=fp, fz=fz,
                                  kn = self.bracket_props.kn,
                                  tension_unity=tension_unity,
                                  governing_capacity=governing_capacity,
                                  reactions_backing=reactions_backing,
                                  reactions_equipment=reactions_equipment)