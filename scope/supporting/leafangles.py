"""Leaf angle distribution functions for SCOPE model.

Translated from: src/supporting/leafangles.m

Based on the two-parameter LIDF model from Verhoef (1998):
"Theory of radiative transfer models applied in optical remote sensing
of vegetation canopies"
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def leafangles(a: float, b: float) -> NDArray[np.float64]:
    """Calculate leaf inclination distribution function (LIDF).

    Computes the LIDF using the two-parameter model from Verhoef.
    The LIDF describes the probability distribution of leaf inclination
    angles in a canopy.

    Args:
        a: LIDF parameter a. Controls the average leaf angle.
           a = -1: erectophile (vertical leaves)
           a = 0: uniform distribution
           a = 1: planophile (horizontal leaves)
        b: LIDF parameter b. Controls the bimodality.
           b = 0: unimodal distribution
           b ≠ 0: bimodal distribution

    Returns:
        LIDF array of shape (13,) representing the probability density
        for each leaf inclination class. The classes are centered at:
        5, 15, 25, 35, 45, 55, 65, 75, 81, 83, 85, 87, 89 degrees.
        Sum of LIDF equals 1.

    Example:
        >>> # Spherical distribution (uniform solid angle)
        >>> lidf = leafangles(-0.35, -0.15)
        >>>
        >>> # Erectophile (vertical leaves)
        >>> lidf = leafangles(-1, 0)
        >>>
        >>> # Planophile (horizontal leaves)
        >>> lidf = leafangles(1, 0)

    Note:
        The 13 inclination classes follow the SAIL convention with
        higher angular resolution near 90° to capture the effect of
        near-vertical leaves on canopy reflectance.
    """
    # Calculate cumulative distribution at class boundaries
    F = np.zeros(13, dtype=np.float64)

    # First 8 classes: 10° to 80° in 10° steps
    for i in range(8):
        theta = (i + 1) * 10  # 10, 20, ..., 80
        F[i] = _dcum(a, b, theta)

    # Next 4 classes: 82°, 84°, 86°, 88°
    for i in range(8, 12):
        theta = 80 + (i - 7) * 2  # 82, 84, 86, 88
        F[i] = _dcum(a, b, theta)

    # Last class: 90° (cumulative = 1 by definition)
    F[12] = 1.0

    # Convert cumulative to probability density
    lidf = np.zeros(13, dtype=np.float64)
    lidf[0] = F[0]
    for i in range(12, 0, -1):
        lidf[i] = F[i] - F[i - 1]

    return lidf


def _dcum(a: float, b: float, theta: float) -> float:
    """Calculate cumulative LIDF at angle theta.

    This is the internal helper function that computes F(θ),
    the cumulative probability up to leaf inclination angle θ.

    Args:
        a: LIDF parameter a
        b: LIDF parameter b
        theta: Leaf inclination angle in degrees

    Returns:
        Cumulative probability F(θ)
    """
    rd = np.pi / 180.0  # degrees to radians

    if a > 1:
        # Special case: approaches cosine distribution
        F = 1 - np.cos(theta * rd)
    else:
        # General case: iterative solution
        eps = 1e-8
        delx = 1.0

        x = 2 * rd * theta
        theta2 = x

        # Iterate to find x that satisfies the implicit equation
        while delx > eps:
            y = a * np.sin(x) + 0.5 * b * np.sin(2 * x)
            dx = 0.5 * (y - x + theta2)
            x = x + dx
            delx = abs(dx)

        F = (2 * y + theta2) / np.pi

    return F


def campbell_lidf(chi_L: float) -> NDArray[np.float64]:
    """Calculate LIDF using Campbell's ellipsoidal distribution.

    An alternative LIDF parameterization using a single parameter.

    Args:
        chi_L: Leaf angle distribution parameter.
               chi_L > 1: planophile (horizontal)
               chi_L = 1: spherical
               chi_L < 1: erectophile (vertical)

    Returns:
        LIDF array of shape (13,)

    Example:
        >>> lidf = campbell_lidf(1.0)  # Spherical distribution
    """
    # Convert Campbell's chi_L to Verhoef's a, b parameters
    # This is an approximation
    if chi_L >= 1:
        a = 2 / chi_L - 1
        b = 0
    else:
        a = 1 - 2 * chi_L
        b = 0

    return leafangles(a, b)


def get_lidf_parameters(distribution: str) -> Tuple[float, float]:
    """Get LIDF parameters for common distribution types.

    Args:
        distribution: Distribution name. One of:
            'planophile': Horizontal leaves (a=1, b=0)
            'erectophile': Vertical leaves (a=-1, b=0)
            'plagiophile': Inclined leaves (a=0, b=-1)
            'extremophile': Extreme angles (a=0, b=1)
            'spherical': Uniform solid angle (a=-0.35, b=-0.15)
            'uniform': Uniform angle distribution (a=0, b=0)

    Returns:
        Tuple of (a, b) LIDF parameters

    Raises:
        ValueError: If distribution name is not recognized

    Example:
        >>> a, b = get_lidf_parameters('spherical')
        >>> lidf = leafangles(a, b)
    """
    distributions = {
        "planophile": (1.0, 0.0),
        "erectophile": (-1.0, 0.0),
        "plagiophile": (0.0, -1.0),
        "extremophile": (0.0, 1.0),
        "spherical": (-0.35, -0.15),
        "uniform": (0.0, 0.0),
    }

    distribution = distribution.lower()
    if distribution not in distributions:
        valid = ", ".join(distributions.keys())
        raise ValueError(f"Unknown distribution '{distribution}'. Valid options: {valid}")

    return distributions[distribution]


def compute_canopy_lidf(canopy) -> NDArray[np.float64]:
    """Compute LIDF from canopy parameters and store in canopy object.

    This is a convenience function that extracts LIDFa and LIDFb from
    a Canopy object, computes the LIDF, and stores it back.

    Args:
        canopy: Canopy dataclass with LIDFa and LIDFb attributes

    Returns:
        LIDF array of shape (13,)

    Side effect:
        Sets canopy.lidf to the computed LIDF
    """
    lidf = leafangles(canopy.LIDFa, canopy.LIDFb)
    canopy.lidf = lidf
    return lidf
