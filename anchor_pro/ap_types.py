
from dataclasses import dataclass
from typing import Protocol, Union, Mapping, runtime_checkable, TypeVar
import numpy as np
from enum import Enum

# import numpy as np
# from typing import Protocol, Dict
#
# class ElementAnalysis(Protocol):
#     def evaluate(
#         self,
#         theta: np.ndarray,              # [n_theta]
#         dof: np.ndarray,                # [n_theta, n_dof]
#     ) -> "ElementResultsBase":          # typed subclass per element
#         ...
#
SeriesOrDict = Union[Mapping[str, object], "pd.Series"]



@runtime_checkable
class Results(Protocol):
    """Protocol for result-like objects that expose a `unity` attribute.
    Makes the protocol usable with isinstance() or issubclass() checks
    if you ever want to assert that something conforms at runtime.
    TypeVar T Keeps the return type consistent with the input list —
    if you pass a list of ConcreteAnchorResults, you’ll get that type back, not just a generic Results."""
    unity: float
    ok: bool
    governing_theta_idx: int

RESULTS_LIKE = TypeVar("RESULTS_LIKE", bound=Results)

class FactorMethod(str, Enum):
    asd = 'ASD'
    lrfd = 'LRFD'
    lrfd_omega = 'LRFD_Omega'


class WallPositions(str, Enum):
    XP = 'X+'
    XN = 'X-'
    YP = 'Y+'
    YN = 'Y-'

class WallNormalVecs(Enum):
    XP = (-1, 0, 0)
    XN = (1, 0, 0)
    YN = (0, -1, 0)
    YP = (0, 1, 0)

