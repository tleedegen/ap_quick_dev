
import numpy as np
from anchor_pro.utilities import (compute_point_displacements,
                                  vertical_point_displacements,
                                  bearing_area_in_compression,
                                  polygon_properties)

#Typing-Related Imports
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from numpy.typing import NDArray
from enum import IntEnum, Enum
from anchor_pro.ap_types import FactorMethod


def effective_indenter_stiffness(area, modulus, poisson_ratio):
    E_eff = modulus / (1 - poisson_ratio ** 2)
    beta = 2 * E_eff / (np.pi * area) ** 0.5
    # beta = 2 * E_eff / 1  # TESTING: trying a "constant" beta to validate impact on speed
    return beta


class LocalDof(IntEnum):
    DX = 0; DY = 1; DZ = 2; RX = 3; RY = 4; RZ = 5

class AnchorStiffnessMode(str, Enum):
    terms = "stiffness_terms"
    forces = "anchor_forces"


@dataclass(frozen=True,slots=True)
class BasePlateReleases:
    """Releases at the connection/attachment point and along directions."""
    mx: bool = False
    my: bool = False
    mz: bool = False
    xp: bool = False
    xn: bool = False
    yp: bool = False
    yn: bool = False
    zp: bool = False
    zn: bool = False

@dataclass(frozen=True, slots=True)
class BasePlateProps:
    """
    Immutable inputs for a baseplate element.

    Notes:
      - θ (load case) axis is the LAST axis everywhere.
      - B is the static conversion matrix mapping global DOFs at (x0,y0,z0)
        to the connection point (xc,yc,zc).
    """
    # Geometry / boundaries
    bearing_boundaries: List[NDArray]

    # Material properties (optional if not used in a particular analysis path)
    E_base: float
    poisson: float

    # Anchor plan coordinates [Global Coordinates] (n_anchor, 2); empty array means "no anchors"
    xy_anchors: NDArray = field(default_factory=lambda: np.empty((0, 2), dtype=float))

    # Inflection/reference point of element (often the “analysis node”) in GLOBAL coordinates
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    # Connection point (where loads are transferred to the plate) in GLOBAL coordiantes
    xc: float = 0.0
    yc: float = 0.0
    zc: float = 0.0

    # Releases
    releases: BasePlateReleases = field(default_factory=BasePlateReleases)

    # Derived fields (populated in __post_init__)
    n_anchor: int = field(init=False)
    B: NDArray = field(init=False)  # static conversion matrix

    def __post_init__(self):
        # Ensure arrays have correct dtype/shape
        xy = np.asarray(self.xy_anchors, dtype=float)
        if xy.ndim != 2 or (xy.size > 0 and xy.shape[1] != 2):
            raise ValueError("xy_anchors must be shape (n_anchor, 2).")

        object.__setattr__(self, "n_anchor", 0 if xy.size == 0 else int(xy.shape[0]))

        # Build the static conversion matrix B using (xc,yc,zc) relative to (x0,y0,z0)
        dx = self.xc - self.x0
        dy = self.yc - self.y0
        dz = self.zc - self.z0

        B = np.array([
            [1.0, 0.0, 0.0, 0.0,    0.0,    0.0],
            [0.0, 1.0, 0.0, 0.0,    0.0,    0.0],
            [0.0, 0.0, 1.0, 0.0,    0.0,    0.0],
            [0.0,  dz,  -dy, 1.0,   0.0,    0.0],
            [-dz,  0.0,  dx, 0.0,   1.0,    0.0],
            [ dy, -dx,  0.0, 0.0,   0.0,    1.0],
        ], dtype=float)

        object.__setattr__(self, "B", B)

@dataclass(frozen=True,slots=True)
class AnchorStiffness:
    shear: float   # k_shear per anchor (N/mm or consistent)
    tension: float # k_tension per anchor

@dataclass(frozen=True,slots=True)
class BasePlateDofs:
    dof_map: NDArray  # (6,)  dof_map[i] is the global dof corresponding to local i
    C: NDArray  # (6,n_dof)
    free_dofs: Tuple[int,...]
    constrained_dofs: Tuple[int,...]

def create_dof_constraints(
    n_dof: int,
    dof_map: NDArray[np.int64],     # (6,)
    props: BasePlateProps
) -> BasePlateDofs:
    """Initializes static (displacement-independent) matrices given the total global dofs and dof_map.
         dof_map is a six element array indicating the index of the global DOFs at each local dofs"""

    """ Returns the constraint matrix (6 x n_dof) relating global DOFs to 6 element DOFs"""

    ''' Presence of vertical releases in both z+ and z- directions results in uncoupling of base plate vertical
    translation as a unique degree of freedom.
    Shear releases do not result in unique degrees of freedom, but rather are handled by modifying the shear
    of the anchors so that resulting plots will show floor plates kinematically constrained to the unit without
    imparting stiffness.'''

    # Sanity
    dof_map = np.asarray(dof_map, dtype=np.int64)
    if dof_map.shape != (6,):
        raise ValueError("dof_map must be shape (6,) of int64")

    if np.any(dof_map < 0) or np.any(dof_map >= n_dof):
        raise ValueError("dof_map entries must be within [0, n_dof).")

    x0, y0, z0 = props.x0, props.y0, props.z0
    xc, yc = props.xc, props.yc

    C = np.zeros((6, n_dof), dtype=float)
    C[LocalDof.DX, 0:6] = [1, 0, 0, 0, z0, -y0]
    C[LocalDof.DY, 0:6] = [0, 1, 0, -z0, 0, x0]

    # Local z-translation row (index 2)
    if dof_map[LocalDof.DZ] != LocalDof.DZ:
        C[LocalDof.DZ, dof_map[LocalDof.DZ]] = 1.0
    else:
        C[LocalDof.DZ, 0:6] = [0, 0, 1, yc, -xc, 0]

    # Local rotations (3,4,5) mapped as identity to their assigned globals
    # todo: verify and revise. There should be coupling for rotation dofs when vertical dof is free on baseplate
    C[LocalDof.RX, dof_map[LocalDof.RX]] = 1.0
    C[LocalDof.RY, dof_map[LocalDof.RY]] = 1.0
    C[LocalDof.RZ, dof_map[LocalDof.RZ]] = 1.0

    # Free vs constrained local indices:
    # Your legacy criterion was "free" if i != dof_map[i] (i.e., remapped to a new global id),
    # and "constrained" if i == dof_map[i] (shares the original base-6 kinematics).
    free_local = tuple(i for i, dof in enumerate(dof_map) if i != dof)
    constrained_local = tuple(i for i, dof in enumerate(dof_map) if i == dof)

    return BasePlateDofs(
        dof_map=dof_map,
        C=C,
        free_dofs=free_local,
        constrained_dofs=constrained_local,
    )

@dataclass(frozen=True, slots=True)
class CompressionZoneState:
    """
    Per-θ compression-zone state. Optional fields are populated only
    when requested by get_compression_zone_properties(...).
    """
    # Basic Properties
    compression_boundaries: Optional[List[NDArray[np.float64]]] = field(default_factory=lambda: np.empty(0))
    areas: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))  # (m,)
    centroids: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty((0, 2)))  # (m, 2)
    beta: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))  # (m,)

    # Inertial Properties
    Ixx: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))  # (m,)
    Iyy: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))  # (m,)
    Ixy: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))  # (m,)

    # Stiffness influence terms (length m)
    k22: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    k23: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    k24: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    k33: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    k34: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    k44: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))

    # Resultants (length m)
    fz: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    mx: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    my: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty(0))
    resultant_centroids: Optional[NDArray[np.float64]] = field(default_factory=lambda: np.empty((0, 2)))  # (m, 2)

@dataclass(frozen=True,slots=True)
class BasePlateResults:
        """
        One sweep of outputs for the baseplate over all θ.

        Shapes (θ last):
          - nodal_forces:        (6, n_theta) forces at analysis node (x0,y0,z0)
          - connection_forces:   (6, n_theta) forces at connection (xc,yc,zc)
          - anchor_forces:       (n_anchor, 6, n_theta) per-anchor [Fx,Fy,Fz,Mx,My,Mz] (if modeled)
          - bearing_resultants:  list of CompressionZoneState objects
        """

        # Reactions
        nodal_forces: NDArray  #(6,n_theta)
        connection_forces: NDArray #(6,n_theta)

        # Anchor Forces
        anchor_forces: NDArray[np.float64]  # (n_anchor, 3, n_theta)

        # Bearing/Compression Zone Outputs:
        compression_zones: List[CompressionZoneState]  # n_theta elements

class BasePlateElement:
    def __init__(self,
                 props: BasePlateProps,
                 dof_props: Optional[BasePlateDofs] = None,
                 anchor_stiffness: Optional[AnchorStiffness]=None):

        self.props = props
        self.dof_props = dof_props
        self.anchor_stiffness=anchor_stiffness
        self.factor_method = FactorMethod.lrfd

    def set_anchor_stiffness(self,anchor_stiffness: AnchorStiffness):
        self.anchor_stiffness = anchor_stiffness

    def get_element_stiffness_matrix(self, u_global: NDArray, initial:bool =False) -> NDArray:
        """ Efficiently assembles the element stiffness matrix without modifying self. """
        C = self.dof_props.C
        u_loc = C @ u_global
        has_anchors = len(self.props.xy_anchors) > 0
        ka = self.anchor_stiffness_matrix(u_loc, initial=initial) if has_anchors else 0.0
        kb = self.bearing_stiffness_matrix(u_loc, initial=initial)

        # Assemble global element stiffness matrix
        k_element = C.T @ (ka + kb) @ C

        return k_element

    def anchor_stiffness_matrix(self, u_local: NDArray, initial: bool = False) -> NDArray:
        """
        Anchor-summed stiffness for a *single* DOF state.
        Returns Ka_sum with shape (6, 6).
        """
        # --- Normalize u to local element DOFs: (6,)
        u = np.asarray(u_local, dtype=float)
        if u.ndim != 1:
            raise ValueError("u must be a 1-D vector for solver use.")

        xy = self.props.xy_anchors
        n_anchor = xy.shape[0]
        if n_anchor == 0:
            return np.zeros((6, 6), dtype=float)

        # Per-anchor stiffness terms
        kx, ky, kz = self.anchor_stiffness_terms(u_local,return_mode=AnchorStiffnessMode.terms)  # (n_anchor, n_theta)

        # Closed-form sums over anchors
        x0, y0, z0 = float(self.props.x0), float(self.props.y0), float(self.props.z0)
        x = xy[:, 0]
        y = xy[:, 1]
        z0_sq = z0 * z0
        dx = (x - x0)[:, None]  # (n_anchor,1) for broadcasting
        dy = (y - y0)[:, None]
        dx2 = dx * dx
        dy2 = dy * dy
        dxdy = dx * dy

        s = lambda a: float(np.sum(a))
        sum_kx = s(kx)
        sum_ky = s(ky)
        sum_kz = s(kz)
        sum_kx_dy = s(kx * dy)
        sum_ky_dx = s(ky * dx)
        sum_kz_dx = s(kz * dx)
        sum_kz_dy = s(kz * dy)
        sum_kz_dx2 = s(kz * dx2)
        sum_kz_dy2 = s(kz * dy2)
        sum_kz_dxdy = s(kz * dxdy)
        sum_ky_dx2 = s(ky * dx2)
        sum_kx_dy2 = s(kx * dy2)

        # --- Build Ka (6x6)
        Ka = np.zeros((6, 6), dtype=float)
        FX, FY, FZ, MX, MY, MZ = 0, 1, 2, 3, 4, 5

        Ka[FX, FX] = sum_kx
        Ka[FX, MY] = Ka[MY, FX] = -z0 * sum_kx
        Ka[FX, MZ] = Ka[MZ, FX] = -sum_kx_dy

        Ka[FY, FY] = sum_ky
        Ka[FY, MX] = Ka[MX, FY] = z0 * sum_ky
        Ka[FY, MZ] = Ka[MZ, FY] = sum_ky_dx

        Ka[FZ, FZ] = sum_kz
        Ka[FZ, MX] = Ka[MX, FZ] = sum_kz_dy
        Ka[FZ, MY] = Ka[MY, FZ] = -sum_kz_dx

        Ka[MX, MX] = sum_kz_dy2 + z0_sq * sum_ky
        Ka[MX, MY] = Ka[MY, MX] = -sum_kz_dxdy
        Ka[MX, MZ] = Ka[MZ, MX] = z0 * sum_ky_dx

        Ka[MY, MY] = sum_kz_dx2 + z0_sq * sum_kx
        Ka[MY, MZ] = Ka[MZ, MY] = z0 * sum_kx_dy

        Ka[MZ, MZ] = sum_ky_dx2 + sum_kx_dy2

        return Ka


    def anchor_stiffness_terms(
        self,
        u_local: np.ndarray,
        *,
        initial: bool = False,
        return_mode: AnchorStiffnessMode = AnchorStiffnessMode.terms,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
            np.ndarray,  # nodal_resultants (6, n_theta)
            np.ndarray,  # anchor_forces (6, n_anchor, n_theta)
        ]:

        u_loc = np.asarray(u_local, dtype=float)
        if u_loc.ndim == 1:
            if u_loc.shape[0] != 6:
                raise ValueError("u_loc must have length 6 when 1D.")
            u_loc = u_loc.reshape(6, 1)
        elif u_loc.ndim == 2:
            if u_loc.shape[0] != 6:
                raise ValueError("u_loc must be (6, n_theta) when 2D.")
        else:
            raise ValueError("u_loc must be either (6,) or (6, n_theta).")

        n_theta = u_loc.shape[1]

        # Geometry
        xy = np.asarray(self.props.xy_anchors, dtype=float)               # (n_anchor, 2)
        n_anchor = xy.shape[0]
        x0, y0, z0 = float(self.props.x0), float(self.props.y0), float(self.props.z0)
        x = xy[:, 0]
        y = xy[:, 1]
        dx = (x - x0)[:, None]                                            # (n_anchor, 1) -> broadcast with (n_anchor, n_theta)
        dy = (y - y0)[:, None]                                            # (n_anchor, 1)
        dz = (0.0 - z0)

        # Points at plate plane (z=0)
        points = np.column_stack((x, y, np.zeros_like(x)))  # (n_anchor, 3)
        delta = compute_point_displacements(points, u_loc, x0=x0, y0=y0, z0=z0)
        if delta.ndim == 2:
            # Single theta came back as 2D; add theta axis
            delta = delta[..., None]
        delta_x = delta[:, 0, :]  # (n_anchor, n_theta)
        delta_y = delta[:, 1, :]  # (n_anchor, n_theta)
        delta_z = delta[:, 2, :]  # (n_anchor, n_theta)

        kv = float(self.anchor_stiffness.shear)
        kt = float(self.anchor_stiffness.tension)

        if initial:
            kx = np.full((n_anchor, n_theta), kv, dtype=float)
            ky = np.full((n_anchor, n_theta), kv, dtype=float)
            kz = np.full((n_anchor, n_theta), kt, dtype=float)
            mask_x = mask_y = mask_z = None
        else:
            rel = self.props.releases
            mask_x = ~((rel.xp & (delta_x >= 0.0)) | (rel.xn & (delta_x < 0.0)))
            mask_y = ~((rel.yp & (delta_y >= 0.0)) | (rel.yn & (delta_y < 0.0)))
            if not (rel.zn and rel.zp):
                mask_z = ((not rel.zp) & (delta_z >= 0.0))
            else:
                mask_z = (delta_z >= 0.0)

            kx = self.anchor_stiffness.shear * mask_x.astype(float)
            ky = self.anchor_stiffness.shear * mask_y.astype(float)
            kz = self.anchor_stiffness.tension * mask_z.astype(float)

        if return_mode == AnchorStiffnessMode.terms:
            return kx, ky, kz

        Fx = kx * delta_x  # (n_anchor, n_theta)
        Fy = ky * delta_y
        Fz = kz * delta_z

        Mx = dy * Fz - dz * Fy  # (n_anchor, n_theta)
        My = dz * Fx - dx * Fz
        Mz = dx * Fy - dy * Fx

        anchor_resultant_contributions = np.stack([Fx, Fy, Fz, Mx, My, Mz], axis=1)  # (6, n_anchor, n_theta)

        anchor_resultants = np.sum(anchor_resultant_contributions, axis=0)  # (6, n_theta) Sum of anchor forces at node
        anchor_forces = anchor_resultant_contributions[:,:3,:]  # (n_anchor,3, n_theta) [Fx, Fy, Fz]

        if return_mode == AnchorStiffnessMode.forces:
            return anchor_resultants, anchor_forces
        else:
            raise Exception("Must specify valid return_mode for anchor_stiffness_terms function")


    def bearing_stiffness_matrix(self, u_local, initial=False):
        """ Computes and returns the bearing stiffness matrix.
         u should be global dof matrix"""


        # Get compression zones (consider caching this if u doesn't change significantly)
        if initial:
            cz = self.get_compression_zone_properties(None, full_bearing=True, return_stiffness_terms=True)
        else:
            cz = self.get_compression_zone_properties(u_local, return_stiffness_terms=True)

        if cz.k22.size == 0:
            return np.zeros((6, 6), dtype=float)

        # Extract stiffness terms assembly
        k22 = cz.k22
        k23 = cz.k23
        k24 = cz.k24
        k33 = cz.k33
        k34 = cz.k34
        k44 = cz.k44

        # Summ Stiffness Terms (over all compression zones)
        S22 = np.sum(k22)
        S23 = np.sum(k23)
        S24 = np.sum(k24)
        S33 = np.sum(k33)
        S34 = np.sum(k34)
        S44 = np.sum(k44)

        Kb = np.zeros((6, 6), dtype=float)
        Kb[2, 2] = S22
        Kb[2, 3] = Kb[3, 2] = S23
        Kb[2, 4] = Kb[4, 2] = S24
        Kb[3, 3] = S33
        Kb[3, 4] = Kb[4, 3] = S34
        Kb[4, 4] = S44

        return Kb

    def get_compression_zone_properties(
            self,
            u_local: Optional[NDArray[np.float64]],
            *,
            full_bearing: bool = False,
            return_basic_props: bool = False,
            return_inertial_props: bool = False,
            return_stiffness_terms: bool = False,
            return_resultants: bool = False,
            eps_bear: float = 1e-8,
    ) -> CompressionZoneState:
        """Returns the geometric properties of the compression zones for given dof values, u"""
        # Identify area of bearing elements that are in compression
        # Loop trough each bearing boundary, then loop through each compression area and assemble list.

        if u_local is None and not full_bearing:
            raise Exception("Must provide dof values or specify 'full_bearing=True' for trial stiffness matrix")

        rel = self.props.releases
        x0, y0, z0 = float(self.props.x0), float(self.props.y0), float(self.props.z0)


        '''Handle Vertical Releases 
            Case 1: negative release only and no moment releases present:
                -> Turn off all bearing
            Case 2: negative release and moment releases are present:
                -> Turn off bearing only if node is negatively displaced
            Case 3: negative and positive releases are present
                -> Plate will receive an independent vertical dof, therefore leave bearing active
        '''
        if (rel.zn and not rel.zp) and not (rel.mx or rel.my):
            release_bearing = True
        elif (rel.zn and not rel.zp) and (vertical_point_displacements(np.array([[x0,y0,z0]]), u_local[0:6])[0]<=0):
            release_bearing = True
        else:
            release_bearing = False

        if release_bearing:
            compression_boundaries = []
        else:
            u_loc = np.array([0, 0, -eps_bear, 0, 0, 0]) if full_bearing else u_local
            compression_boundaries = [
                cb
                for boundary in self.props.bearing_boundaries
                for cb in bearing_area_in_compression(boundary, u_loc, x0=x0, y0=y0)]

        # Get stiffness and inertial properties for compression areas
        m = len(compression_boundaries)


        need_props = (m>0)
        need_stiffness = (m > 0) and (return_stiffness_terms or return_resultants)

        areas = np.empty(m, dtype=float)
        centroids = np.empty((m, 2), dtype=float)
        Ixx_list = np.empty(m, dtype=float)
        Iyy_list = np.empty(m, dtype=float)
        Ixy_list = np.empty(m, dtype=float)
        beta_list = np.empty(m, dtype=float)

        if need_props:
            for i, vertices in enumerate(compression_boundaries):
                A, ctr, Ixx, Iyy, Ixy = polygon_properties(vertices)
                beta = effective_indenter_stiffness(A, self.props.E_base, self.props.poisson)
                areas[i] = A
                centroids[i] = ctr
                Ixx_list[i] = Ixx
                Iyy_list[i] = Iyy
                Ixy_list[i] = Ixy
                beta_list[i] = (beta / 10.0) if full_bearing else beta  # heuristic to reduce stiffness for initial stiffness matrix

        if need_stiffness:
            xbar = centroids[:,0]
            ybar = centroids[:,1]
            dx = centroids[:, 0] - x0
            dy = centroids[:, 1] - y0
            A = areas
            beta = beta_list
            Ixx, Iyy, Ixy = Ixx_list, Iyy_list, Ixy_list

            k22 = beta * A
            k23 = beta * dy * A
            k24 = -beta * dx * A
            k33 = beta * (Ixx - 2.0 * y0 * ybar * A + (y0 ** 2) * A)
            k34 = -beta * (Ixy - y0 * xbar * A - x0 * ybar * A + x0 * y0 * A)
            k44 = beta * (Iyy - 2.0 * x0 * xbar * A + (x0 ** 2) * A)

        cz = CompressionZoneState()

        if return_basic_props:
            object.__setattr__(cz,'compression_boundaries',compression_boundaries)
            object.__setattr__(cz,'areas',areas)
            object.__setattr__(cz, 'centroids', centroids)
            object.__setattr__(cz, 'beta', beta_list)

        if return_inertial_props:
            if m == 0:
                object.__setattr__(cz, "Ixx", np.empty((0,), float))
                object.__setattr__(cz, "Iyy", np.empty((0,), float))
                object.__setattr__(cz, "Ixy", np.empty((0,), float))
            else:
                object.__setattr__(cz, "Ixx", Ixx_list)
                object.__setattr__(cz, "Iyy", Iyy_list)
                object.__setattr__(cz, "Ixy", Ixy_list)

        if return_stiffness_terms:
            if m == 0:
                # Return empty arrays if explicitly requested but no zones
                object.__setattr__(cz, "k22", np.empty((0,), float))
                object.__setattr__(cz, "k23", np.empty((0,), float))
                object.__setattr__(cz, "k24", np.empty((0,), float))
                object.__setattr__(cz, "k33", np.empty((0,), float))
                object.__setattr__(cz, "k34", np.empty((0,), float))
                object.__setattr__(cz, "k44", np.empty((0,), float))
            else:
                object.__setattr__(cz, "k22", k22)
                object.__setattr__(cz, "k23", k23)
                object.__setattr__(cz, "k24", k24)
                object.__setattr__(cz, "k33", k33)
                object.__setattr__(cz, "k34", k34)
                object.__setattr__(cz, "k44", k44)

        if return_resultants:
            if m == 0:
                object.__setattr__(cz, "fz", np.empty((0,), float))
                object.__setattr__(cz, "mx", np.empty((0,), float))
                object.__setattr__(cz, "my", np.empty((0,), float))
                object.__setattr__(cz, "resultant_centroids", np.empty((0, 2), float))
            else:
                uz, urx, ury = float(u_local[2]), float(u_local[3]), float(u_local[4])
                fz = k22 * uz + k23 * urx + k24 * ury
                mx = k23 * uz + k33 * urx + k34 * ury
                my = k24 * uz + k34 * urx + k44 * ury

                with np.errstate(divide="ignore", invalid="ignore"):
                    cx = np.where(fz != 0.0, -my / fz + x0, np.nan)
                    cy = np.where(fz != 0.0, mx / fz + y0, np.nan)
                rc = np.stack([cx, cy], axis=-1)

                object.__setattr__(cz, "fz", fz)
                object.__setattr__(cz, "mx", mx)
                object.__setattr__(cz, "my", my)
                object.__setattr__(cz, "resultant_centroids", rc)

        return cz

    def compute_bearing_resultants_over_theta(
            self,
            u_local: NDArray,  # (n_dof, n_theta)
            return_states: bool = True,
    )->Tuple[List[CompressionZoneState], NDArray, NDArray]:
        """
        Vectorized bearing resultants over all θ.
        Returns:
          - bearing_resultants: (6, n_theta) with only [2]=Fz, [3]=Mx, [4]=My filled
          - bearing_centroid:   (2, n_theta) aggregate resultant location (x,y)
          - snapshots:          list of CompressionZoneSnapshot (len=n_theta) if keep_snapshots else None
        """
        # ---- Map to local DOFs: (6, n_theta)
        u_local = np.asarray(u_local, dtype=float)
        if u_local.ndim != 2:
            raise ValueError("U must be 2-D: (N, n_theta)")

        n_theta = u_local.shape[1]
        x0, y0 = float(self.props.x0), float(self.props.y0)

        bearing_resultants = np.zeros((6, n_theta), dtype=float)  # Fz,Mx,My used
        bearing_centroid = np.full((2, n_theta), np.nan, dtype=float)
        cz_states: Optional[List[CompressionZoneState]] = [] if return_states else None

        # ---- θ loop for clipping; vectorize within
        for t in range(n_theta):
            u_t = u_local[:, t]

            # Get CZ state for this θ (reuses your clipping + property code)
            cz = self.get_compression_zone_properties(u_t,
                                                      return_basic_props=True,
                                                      return_resultants=True)

            # Aggregate (for computing connection reactions
            Fz = float(np.sum(cz.fz))
            Mx = float(np.sum(cz.mx))
            My = float(np.sum(cz.my))

            bearing_resultants[2, t] = Fz
            bearing_resultants[3, t] = Mx
            bearing_resultants[4, t] = My

            if Fz != 0.0:
                bearing_centroid[0, t] = -My / Fz + x0
                bearing_centroid[1, t] = Mx / Fz + y0

            if return_states:
                cz_states.append(cz)

        return cz_states, bearing_resultants, bearing_centroid


    def evaluate(self,u: NDArray) -> BasePlateResults:
        # Get Anchor Resultants
        C = self.dof_props.C
        u_local = C @ u
        if len(self.props.xy_anchors)>0:
            anchor_resultants, anchor_forces = self.anchor_stiffness_terms(u_local,return_mode=AnchorStiffnessMode.forces)
        else:
            anchor_resultants = 0.0
            anchor_forces = 0.0

        # Get bearing Resultants
        cz_states, bearing_resultants, bearing_centroid = self.compute_bearing_resultants_over_theta(u_local)

        nodal_forces = anchor_resultants + bearing_resultants
        connection_forces = -self.props.B@nodal_forces  # Negative to give reactions on sms, not plate

        return BasePlateResults(nodal_forces=nodal_forces,
                                connection_forces=connection_forces,
                                anchor_forces=anchor_forces,
                                compression_zones = cz_states)
