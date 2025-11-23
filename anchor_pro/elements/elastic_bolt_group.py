from dataclasses import dataclass
from typing import Tuple, Optional
from numpy.typing import NDArray
import numpy as np

Vec3 = Tuple[float, float, float]

@dataclass(frozen=True, slots=True)
class InertialProps:
    dx: NDArray
    dy: NDArray
    Ixx: float  # about local-x axis
    Iyy: float  # about local-y axis
    Ixy: float  # coupling (often 0 with symmetric axes)
    Ip: float  # polar = Ixx + Iyy (for in-plane shear distribution)

def compute_inertial_props(xy_anchors: NDArray, rotation_point: NDArray) -> InertialProps:
    if xy_anchors.ndim != 2 or xy_anchors.shape[1] != 2:
        raise ValueError("xy_anchors must be (n_a, 2)")
    if rotation_point.shape != (2,):
        raise ValueError("rotation_point must be shape (2,)")

    dx = xy_anchors[:,0] - rotation_point[0]
    dy = xy_anchors[:,1] - rotation_point[1]

    Ixx = sum(dy**2)
    Iyy = sum(dx**2)
    Ixy = sum(dx*dy)
    Ip = Ixx + Iyy

    return InertialProps(dx=dx, dy=dy, Ixx=Ixx,Iyy=Iyy,Ixy=Ixy,Ip=Ip)


@dataclass(frozen=True, slots=True)
class ElasticBoltGroupProps:
    # Bounding rectangle (local coords), used for edge distances and x/y origins
    w: float
    h: float

    # Fastener coordinates in the element's local (p, z) plane.
    # Shape: (n_anchor, 2) with columns [p, z]
    xy_anchors: NDArray[np.float64]  # (n_a, 2); NOTE: stores [p, z] despite the name

    # Precomputed “bolt group” section properties in local axes (about rectangle origin)
    plate_centroid_XYZ: NDArray[np.float64]  # (3,) global centroid, (x,y,z)
    # Global placement/orientation
    local_x: NDArray[np.float64]  # (3,) unit vector in global coords
    local_y: NDArray[np.float64]  # (3,) unit vector in global coords
    local_z: NDArray[np.float64]  # (3,) unit vector in global coords


    # Derived
    n_anchors: int = 0
    anchor_centroid: Optional[NDArray[np.float64]] = None

    # Global to local transformation matrix [local is M @ global]
    global_to_local_transformation: NDArray[np.float64] = None  # (3,3)

    # Inertial properties
    inert_props_cent: Optional[InertialProps] = None # Inertial properties w.r.t. bolt group centroid
    inert_props_xp: Optional[InertialProps] = None  # (w.r.t. y-edge for positive moment about x-x axis)
    inert_props_yp: Optional[InertialProps] = None  # (w.r.t. x-edge for positive moment about y-y axis)
    inert_props_xn: Optional[InertialProps] = None  # (w.r.t. y-edge for negative moment about x-x axis)
    inert_props_yn: Optional[InertialProps] = None  # (w.r.t. x-edge for negative moment about y-y axis)

    def __post_init__(self):
        #Number of anchors
        n = self.xy_anchors.shape[0]
        object.__setattr__(self,'n_anchors',n)

        # Anchor Centroids
        cent = np.mean(self.xy_anchors, axis=0)
        if (abs(cent[0]) > self.w/2) or (abs(cent[1]) > self.h/2):
            raise Exception("Fastener group centroid must be within bounding rectangle.")
        object.__setattr__(self,'anchor_centroid',cent)

        # Global to local transformation matrix
        lx = np.asarray(self.local_x, dtype=np.float64)
        ly = np.asarray(self.local_y, dtype=np.float64)
        lz = np.asarray(self.local_z, dtype=np.float64)

        # Normalize local axes (defensive)
        def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
            n = np.linalg.norm(v)
            if n == 0.0:
                raise ValueError("Local axis has zero length.")
            return v / n

        lx, ly, lz = _unit(lx), _unit(ly), _unit(lz)
        object.__setattr__(self, "local_x", lx)
        object.__setattr__(self, "local_y", ly)
        object.__setattr__(self, "local_z", lz)

        R = np.vstack([lx, ly, lz])
        object.__setattr__(self, "global_to_local_transformation",R)

        # Centroidal Inertial Properties
        ic = compute_inertial_props(self.xy_anchors,self.anchor_centroid)
        object.__setattr__(self,'inert_props_cent',ic)

        # Inertial Props for bending about positive X
        y_edge = -self.h / 2.0
        ix = compute_inertial_props(self.xy_anchors,np.array([self.anchor_centroid[0],y_edge]))
        object.__setattr__(self,'inert_props_xp',ix)

        # Inertial Props for bending about negative X
        y_edge = self.h / 2.0
        ix = compute_inertial_props(self.xy_anchors, np.array([self.anchor_centroid[0], y_edge]))
        object.__setattr__(self, 'inert_props_xn', ix)

        # Inertial Props for bending about positive Y
        x_edge = self.w / 2.0
        iy = compute_inertial_props(self.xy_anchors, np.array([x_edge, self.anchor_centroid[1]]))
        object.__setattr__(self, 'inert_props_yp', iy)

        # Inertial Props for bending about negative Y
        x_edge = -self.w / 2.0
        iy = compute_inertial_props(self.xy_anchors, np.array([x_edge, self.anchor_centroid[1]]))
        object.__setattr__(self, 'inert_props_yn', iy)

@dataclass(frozen=True, slots=True)
class ElasticBoltGroupResults:
    # Centroid (backing) resultant forces, local coordinates
    # Shape: (6, n_theta) ordered [N, Vp, Vz, Mx, My, T]
    centroid_forces_loc: NDArray  # (6, n_theta)

    # Anchor forces (at each anchor), local coords
    # Shape: (n_anchor, 3, n_theta) components [n, vp, vz]
    anchor_forces_loc: NDArray  # (n_a, 3, n_t)


def calculate_bolt_group_forces(
    N: NDArray[np.float64],
    Vx: NDArray[np.float64],
    Vy: NDArray[np.float64],
    Mx: NDArray[np.float64],
    My: NDArray[np.float64],
    T:  NDArray[np.float64],   # Applied resultants at bolt-group reference
    n_anchors: int,
    inert_c: InertialProps,                   # centroid-origin inertials (must expose Ixx,Iyy,Ixy,Ip, dx,dy)
    inert_xp: Optional[InertialProps] = None,  # inertials for +Mx term (same interface)
    inert_xn: Optional[InertialProps] = None,  # inertials for -Mx term (same interface)
    inert_yp: Optional[InertialProps] = None,  # inertials for +My term (same interface)
    inert_yn: Optional[InertialProps] = None,  # inertials for -My term (same interface)
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Returns
    -------
    anchor_forces_loc : (n_anchor, 3, n_theta) [n, vx, vy]
    """
    # Ensure arrays and shapes
    N  = np.asarray(N,  dtype=np.float64).reshape(-1)
    Vx = np.asarray(Vx, dtype=np.float64).reshape(-1)
    Vy = np.asarray(Vy, dtype=np.float64).reshape(-1)
    Mx = np.asarray(Mx, dtype=np.float64).reshape(-1)
    My = np.asarray(My, dtype=np.float64).reshape(-1)
    T  = np.asarray(T,  dtype=np.float64).reshape(-1)

    n_t = N.shape[0]
    n_a = int(n_anchors)
    if n_a <= 0:
        raise ValueError("n_anchors must be > 0")

    # Convert to row vectors for easy term multiplication (1, n_t)
    N_b  = N[None, :]
    Vx_b = Vx[None, :]
    Vy_b = Vy[None, :]
    Mx_b = Mx[None, :]
    My_b = My[None, :]
    T_b  = T[None, :]

    # Select inertial packs
    ixp = inert_xp if inert_xp is not None else inert_c
    iyp = inert_yp if inert_yp is not None else inert_c
    ixn = inert_xn if inert_xn is not None else inert_c
    iyn = inert_yn if inert_yn is not None else inert_c
    ic = inert_c

    # Pull dx, dy as (n_a, 1)
    def _col(a) -> NDArray[np.float64]:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        if a.shape != (n_a, 1):
            raise ValueError(f"dx/dy must be shape ({n_a}, 1); got {a.shape}")
        return a

    dx_c, dy_c = _col(ic.dx), _col(ic.dy)
    dx_xp, dy_xp = _col(ixp.dx), _col(ixp.dy)
    dx_xn, dy_xn = _col(ixn.dx), _col(ixn.dy)
    dx_yp, dy_yp = _col(iyp.dx), _col(iyp.dy)
    dx_yn, dy_yn = _col(iyn.dx), _col(iyn.dy)

    # -----------------------
    # Normal term: equal share
    # -----------------------
    normal_term = N_b / max(n_a, 1)             # (1, n_t), relies on broadcasting later

    # -----------------------
    # Mx bending term
    # -----------------------

    sign_mask = (Mx_b >= 0)
    Ixx = np.where(sign_mask, ixp.Ixx, ixn.Ixx).astype(float)  # (n_t,)
    Iyy = np.where(sign_mask, ixp.Iyy, ixn.Iyy).astype(float)  # (n_t,)
    Ixy = np.where(sign_mask, ixp.Ixy, ixn.Ixy).astype(float)  # (n_t,)
    denom = Ixx * Iyy - Ixy ** 2  # (n_t,)
    dx = np.where(sign_mask, dx_xp, dx_xn)
    dy = np.where(sign_mask, dy_xp, dy_xn)


    if (abs(Ixy) <= eps).all():
        # Decoupled case -> y * (Mx / Ixx)
        if (abs(Ixx) > eps).all():
            mx_term = dy * (Mx_b / Ixx)     # (n_a, n_t)
        else:
            mx_term = np.zeros((n_a, n_t))
    else:
        if (abs(denom) > eps).all():
            # Coupled case -> (Mx/denom)*(Iyy*dy - Ixy*dx)
            mx_term = (Mx_b / denom) * (Iyy * dy - Ixy * dx)
        else:
            mx_term = np.zeros((n_a, n_t))

    # -----------------------
    # My bending term
    # -----------------------
    sign_mask = (Mx_b >= 0)
    Ixx = np.where(sign_mask, iyp.Ixx, iyn.Ixx).astype(float)  # (n_t,)
    Iyy = np.where(sign_mask, iyp.Iyy, iyn.Iyy).astype(float)  # (n_t,)
    Ixy = np.where(sign_mask, iyp.Ixy, iyn.Ixy).astype(float)  # (n_t,)
    denom = Ixx * Iyy - Ixy ** 2  # (n_t,)
    dx = np.where(sign_mask, dx_yp, dx_yn)
    dy = np.where(sign_mask, dy_yp, dy_yn)


    if (abs(Ixy) <= eps).any():
        # Decoupled case -> (+) x * (My / Iyy)
        # (Sign preserved to match your current convention.)
        if (abs(Iyy) > eps).any():
            my_term = dx * (My_b / Iyy)     # (n_a, n_t)
        else:
            my_term = np.zeros((n_a, n_t))
    else:
        if (abs(denom) > eps).any():
            # Coupled case -> (My/denom)*(Ixx*dx - Ixy*dy)
            my_term = (My_b / denom) * (Ixx * dx - Ixy * dy)
        else:
            my_term = np.zeros((n_a, n_t))

    # Combine axial (normal) contributions
    n_force = normal_term + mx_term + my_term   # (n_a, n_t) via broadcasting

    # -----------------------
    # In-plane shear + torsion
    # -----------------------
    Ip_c = float(ic.Ip)
    if abs(Ip_c) <= eps:
        vx_force = np.ones((n_a, 1)) @ (Vx_b / max(n_a, 1))
        vy_force = np.ones((n_a, 1)) @ (Vy_b / max(n_a, 1))
    else:
        # Classic rigid distribution with torsion: +/- T * (offset) / Ip
        vx_force = np.ones((n_a, 1)) @ (Vx_b / n_a) - (T_b * dy_c) / Ip_c
        vy_force = np.ones((n_a, 1)) @ (Vy_b / n_a) + (T_b * dx_c) / Ip_c

    # Pack to (n_a, 3, n_t)  -> [n, vx, vy]
    anchor_forces_loc = np.stack([n_force, vx_force, vy_force], axis=1)

    return anchor_forces_loc