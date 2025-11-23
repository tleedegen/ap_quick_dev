from dataclasses import dataclass
from typing import Tuple
import cmath
import math

@dataclass
class Quadratic_Formula:
    a : float
    b : float
    c : float

    def solve(self) -> Tuple[complex, complex]:
        d = cmath.sqrt(self.b**2 - 4*self.a*self.c)
        r1 = (-self.b + d) / (2*self.a)
        r2 = (-self.b - d) / (2*self.a)
        return r1, r2


@dataclass
class Pythagorean_Theorem:
    a: float
    b: float

    def solve(self) -> float:
        return math.sqrt(self.a**2 + self.b**2)
