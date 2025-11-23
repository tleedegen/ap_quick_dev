
from dataclasses import dataclass
from enum import Enum
import numpy as np
from anchor_pro.ap_types import FactorMethod
from typing import Optional, Tuple
from numpy.typing import NDArray
import pandas as pd


class SMSDisallowedCombination(ValueError): pass


class SMSCatalog:
    REQUIRED = {"sms_size", "condition", "fy", "gauge", "shear", "tension"}
    def __init__(self, df_sms: pd.DataFrame):
        # Verify that input catalog from has required columns
        missing = SMSCatalog.REQUIRED - set(df_sms.columns)
        if missing:
            raise ValueError(f"SMS table missing columns: {sorted(missing)}")

        # Remove rows for non-applicable combinations of gauge and fy
        df = df_sms.copy()
        df = df.dropna(subset=["shear", "tension"])
        df = df[(df["shear"] > 0) & (df["tension"] > 0)]

        # Normalize keys
        df["sms_size"] = df["sms_size"].astype(str).str.strip()
        df["condition"] = df["condition"].astype(str).str.strip()
        df["fy"] = df["fy"].astype(float)
        df["gauge"] = df["gauge"].astype(float)

        # Convert to a dictionary for speedy lookup
        self._caps: dict[Tuple[str, str, float, float], Tuple[float, float]] = {
            (r.sms_size, r.condition, r.fy, r.gauge): (r.tension, r.shear)
            for r in df.itertuples(index=False)
        }

    def get(self, size: str, condition: str, fy: float, gauge: float) -> Tuple[float, float]:
        key = (size.strip(), condition.strip(), float(fy), float(gauge))
        try:
            return self._caps[key]
        except KeyError:
            raise SMSDisallowedCombination(
                f"Disallowed SMS combination: size={key[0]}, condition={key[1]}, fy={key[2]}, gauge={key[3]}"
            )


class SMSCondition(str, Enum):
    """Conditions related to the SMS capacity Tables from OPD-001 ."""
    METAL_ON_METAL = "Condition 1"
    GYP_1_LAYER    = "Condition 2"
    GYP_2_LAYERS   = "Condition 3"
    PRYING         = "Condition 4"


@dataclass(frozen=True, slots=True)
class SMSCaps:
    cap_T: float     # tension capacity
    cap_Vx: float    # shear capacity for X direction condition
    cap_Vy: float    # shear capacity for Y direction condition


@dataclass(frozen=True, slots=True)
class SMSProps:
    fy: float                  # ksi or MPa – document units!
    gauge: float               # sheet thickness gauge

    # condition to use for each shear direction:
    condition_x: SMSCondition
    condition_y: SMSCondition


def get_sms_caps(cat: SMSCatalog,screw_size, props: SMSProps) -> SMSCaps:
    cap_T, cap_Vx = cat.get(screw_size, props.condition_x.value, props.fy, props.gauge)
    _, cap_Vy     = cat.get(screw_size, props.condition_y.value, props.fy, props.gauge)
    return SMSCaps(cap_T, cap_Vx, cap_Vy)


@dataclass(frozen=True, slots=True)
class SMSResults:
    # Demands per theta & anchor
    tension_demand: NDArray[np.float32]   # [n_anchor, n_theta]
    shear_x_demand: NDArray[np.float32]   # [n_anchor, n_theta]
    shear_y_demand: NDArray[np.float32]   # [n_anchor, n_theta]

    # Capacities (broadcasted to [n_theta, n_anchor] for uniformity)
    tension_capacity: float # [n_anchor, n_theta]
    shear_x_capacity: float # [n_anchor, n_theta]
    shear_y_capacity: float # [n_anchor, n_theta]

    # Utilization ratios (same shape)
    tension_unity: NDArray[np.float32]    # [n_anchor, n_theta]
    shear_x_unity: NDArray[np.float32]    # [n_anchor, n_theta]
    shear_y_unity: NDArray[np.float32]    # [n_anchor, n_theta]
    shear_unity: NDArray[np.float32]      # [n_anchor, n_theta]
    unities: NDArray
    unity: float
    ok: NDArray[np.bool_]                 # [n_anchor, n_theta]

    governing_anchor_idx: int
    governing_theta_idx: int

    # Optional intermediates for reporting
    # intermediates: Dict[str, np.ndarray] = field(default_factory=dict)

    # Optional: echo props used for this run
    # props: SMSProps


class SMSAnchors:
    __slots__ = ("elem_id", "props","screw_size", "caps", "factor_method")

    def __init__(
            self, props: Optional[SMSProps],
            screw_size: Optional[str] = None,
            caps: Optional[SMSCaps]=None):
        self.props = props
        self.screw_size = screw_size  # e.g., "#10", "12-14"
        self.caps = caps
        self.factor_method = FactorMethod.asd

    def set_screw_size(self,catalog:SMSCatalog,
                  sms_size='No. 10'):
        """
        Assigns a new SMSProps and corresponding SMSCaps to this element.
        Call this before evaluation if product selection changes between runs.
        """

        # self.props = SMSProps(fy=fy,gauge=gauge,screw_size=screw_size,condition_x=condition_x,condition_y=condition_y)
        self.screw_size=sms_size
        self.caps = get_sms_caps(catalog ,sms_size, self.props)

    def evaluate(self, forces: NDArray[np.float32]) -> SMSResults:
        """
        forces: [n_theta, n_anchor, 3] as [T, Vx, Vy]
        theta:  [n_theta]
        """
        # Verify Props have been set for object
        if self.caps is None or self.props is None:
            raise RuntimeError("SMSAnchors not configured: call set_props(...) or pass props/caps in constructor.")

        # Extract
        f = np.asarray(forces, dtype=np.float32)
        if f.ndim != 3 or f.shape[1] != 3:
            raise ValueError("forces must be shape [n_theta, n_anchor, 3] = [T, Vx, Vy]")

        T, Vx, Vy = f[:,0, :], f[:, 1, :], f[:, 2, :]

        capT = self.caps.cap_T  # scalar
        capVx = self.caps.cap_Vx  # scalar
        capVy = self.caps.cap_Vy  # scalar

        with np.errstate(divide="ignore", invalid="ignore"):
            unity_T  = (np.clip(T ,0, None) / capT).astype(np.float32, copy=False)
            unity_Vx = (np.abs(Vx) / capVx).astype(np.float32, copy=False)
            unity_Vy = (np.abs(Vy) / capVy).astype(np.float32, copy=False)

        shear_unity = _combine_shear(unity_Vx, unity_Vy)
        unities = shear_unity + unity_T
        unity = unities.max()
        max_anchor, max_theta = np.unravel_index(np.argmax(unities, axis=None), unities.shape)
        ok = (unity < 1.0)

        return SMSResults(
            tension_demand=T, shear_x_demand=Vx, shear_y_demand=Vy,
            tension_capacity=capT, shear_x_capacity=capVx, shear_y_capacity=capVy,
            tension_unity=unity_T, shear_x_unity=unity_Vx, shear_y_unity=unity_Vy,
            shear_unity=shear_unity,unities=unities, unity=unity, ok=ok,
            governing_anchor_idx=int(max_anchor), governing_theta_idx=int(max_theta)
        )

def _combine_shear(unity_vx: NDArray[np.float32], unity_vy: NDArray[np.float32]) -> NDArray[np.float32]:
    """ Function to combine shear unities"""
    # Option A: Euclidean
    return np.hypot(unity_vx, unity_vy)
    # Option B: Max
    # return np.maximum(unity_Vx, unity_Vy)
    # Option C: Weighted, etc.