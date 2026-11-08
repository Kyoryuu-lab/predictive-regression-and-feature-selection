"""
==============================================================================
Module:      fuzzy_operations.py
Description: Arithmetic operations on triangular fuzzy numbers (Addition,
             Subtraction, Multiplication) and piecewise membership visualization.
==============================================================================
"""

import numpy as np
from typing import Tuple


def triangular_membership(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Berechnet die stueckweise lineare Dreiecks-Zugehoerigkeitsfunktion."""
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0.0)


def add_triangular_fuzzy(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Addition zweier Dreiecks-Fuzzy-Zahlen: (a1+b1, a2+b2, a3+b3)."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def subtract_triangular_fuzzy(c: Tuple[float, float, float], d: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Subtraktion zweier Dreiecks-Fuzzy-Zahlen: (c1-d3, c2-d2, c3-d1)."""
    return (c[0] - d[2], c[1] - d[1], c[2] - d[0])


def multiply_triangular_fuzzy(e: Tuple[float, float, float], f: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Multiplikation zweier Dreiecks-Fuzzy-Zahlen: (e1*f1, e2*f2, e3*f3)."""
    return (e[0] * f[0], e[1] * f[1], e[2] * f[2])
