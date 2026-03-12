"""Aerodynamic and boundary layer resistance calculations.

Translated from: src/fluxes/resistances.m

Calculates resistances for heat and mass transfer in the soil-vegetation-
atmosphere system using Monin-Obukhov similarity theory.

References:
    Wallace, J. S., & Verhoef, A. (2000). Modelling interactions in
    mixed-plant communities: light, water and carbon dioxide.
    In Leaf Development and Canopy Growth, Sheffield Academic Press.

    Paulson, C. A. (1970). The mathematical representation of wind speed
    and temperature profiles in the unstable atmospheric surface layer.
    Journal of Applied Meteorology, 9, 857-861.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..constants import CONSTANTS


@dataclass
class ResistanceOutput:
    """Output from resistance calculations.

    Attributes:
        ustar: Friction velocity [m s-1]
        raa: Aerodynamic resistance above canopy [s m-1]
        rawc: Total resistance within canopy (canopy leaves) [s m-1]
        raws: Total resistance within canopy (soil) [s m-1]
        rai: Aerodynamic resistance in inertial sublayer [s m-1]
        rar: Aerodynamic resistance in roughness sublayer [s m-1]
        rac: Aerodynamic resistance in canopy layer [s m-1]
        rws: Aerodynamic resistance within canopy to soil [s m-1]
        uz0: Wind speed at roughness height [m s-1]
        Kh: Diffusivity for heat [m2 s-1]
    """

    ustar: Union[float, NDArray[np.float64]] = 0.0
    raa: Union[float, NDArray[np.float64]] = 0.0
    rawc: Union[float, NDArray[np.float64]] = 0.0
    raws: Union[float, NDArray[np.float64]] = 0.0
    rai: Union[float, NDArray[np.float64]] = 0.0
    rar: Union[float, NDArray[np.float64]] = 0.0
    rac: Union[float, NDArray[np.float64]] = 0.0
    rws: Union[float, NDArray[np.float64]] = 0.0
    uz0: Union[float, NDArray[np.float64]] = 0.0
    Kh: Union[float, NDArray[np.float64]] = 0.0


def psim(
    z: float,
    L: float,
    unstable: bool,
    stable: bool,
    x: float,
) -> float:
    """Stability correction function for momentum (Paulson, 1970).

    Args:
        z: Height above displacement [m]
        L: Obukhov length [m]
        unstable: True if unstable conditions
        stable: True if stable conditions
        x: Stability parameter (1-16*z/L)^0.25

    Returns:
        Stability correction for momentum
    """
    pm = 0.0
    if unstable:
        pm = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
    elif stable:
        pm = -5 * z / L
    return pm


def psih(
    z: float,
    L: float,
    unstable: bool,
    stable: bool,
    x: float,
) -> float:
    """Stability correction function for heat (Paulson, 1970).

    Args:
        z: Height above displacement [m]
        L: Obukhov length [m]
        unstable: True if unstable conditions
        stable: True if stable conditions
        x: Stability parameter

    Returns:
        Stability correction for heat
    """
    ph = 0.0
    if unstable:
        ph = 2 * np.log((1 + x ** 2) / 2)
    elif stable:
        ph = -5 * z / L
    return ph


def phstar(
    z: float,
    zR: float,
    d: float,
    L: float,
    stable: bool,
    unstable: bool,
    x: float,
) -> float:
    """Modified stability correction for roughness sublayer.

    Args:
        z: Height [m]
        zR: Roughness sublayer height [m]
        d: Displacement height [m]
        L: Obukhov length [m]
        stable: True if stable conditions
        unstable: True if unstable conditions
        x: Stability parameter

    Returns:
        Modified stability correction
    """
    phs = 0.0
    if unstable:
        phs = (z - d) / (zR - d) * (x ** 2 - 1) / (x ** 2 + 1)
    elif stable:
        phs = -5 * z / L
    return phs


def resistances(
    constants,
    soil,
    canopy,
    meteo,
) -> ResistanceOutput:
    """Calculate aerodynamic and boundary resistances.

    Translated from MATLAB resistances.m

    Implements the Wallace & Verhoef (2000) resistance network for
    soil-vegetation-atmosphere exchange.

    Args:
        constants: Physical constants structure with kappa
        soil: Soil structure with rbs
        canopy: Canopy structure with Cd, LAI, rwc, zo, d, hc
        meteo: Meteo structure with z, u, L

    Returns:
        ResistanceOutput with all resistance components
    """
    # Extract parameters from structures
    kappa = constants.kappa
    Cd = canopy.Cd if hasattr(canopy, 'Cd') else 0.3
    LAI = canopy.LAI
    rwc = canopy.rwc if hasattr(canopy, 'rwc') else 2.0
    z0m = canopy.zo if hasattr(canopy, 'zo') else 0.1 * canopy.hc
    d = canopy.d if hasattr(canopy, 'd') else 0.7 * canopy.hc
    h = canopy.hc
    z = meteo.z if hasattr(meteo, 'z') else 2.0
    u = max(0.3, meteo.u)
    L = meteo.L if hasattr(meteo, 'L') else 1e6
    rbs = soil.rbs if hasattr(soil, 'rbs') else 10.0

    # Derived parameters
    zr = 2.5 * h  # Top of roughness sublayer [m]
    n = Cd * LAI / (2 * kappa ** 2)  # Wind extinction coefficient

    # Stability conditions (L=None means neutral)
    if L is None or abs(L) > 1e6:
        # Neutral conditions (very large |L| or unspecified)
        unstable = False
        stable = False
        L = 1e10  # Use large value to avoid division issues
    else:
        unstable = L < 0 and L > -500
        stable = L > 0 and L < 500

    # Stability parameter
    x = (1 - 16 * z / L) ** 0.25 if unstable else 1.0

    # Stability corrections at different heights
    pm_z = psim(z - d, L, unstable, stable, x)
    ph_z = psih(z - d, L, unstable, stable, x)
    pm_h = psim(h - d, L, unstable, stable, x)
    ph_zr = psih(zr - d, L, unstable, stable, x) if z >= zr else ph_z
    phs_zr = phstar(zr, zr, d, L, stable, unstable, x)
    phs_h = phstar(h, zr, d, L, stable, unstable, x)

    # Friction velocity (Eq. 30)
    ustar = max(0.001, kappa * u / (np.log((z - d) / z0m) - pm_z))

    # Diffusivity for heat (Eq. 35)
    Kh = kappa * ustar * (zr - d)
    if unstable:
        Kh = Kh * (1 - 16 * (h - d) / L) ** 0.5
    elif stable:
        Kh = Kh * (1 + 5 * (h - d) / L) ** (-1)

    # Wind speed at height h and z0m
    uh = max(0.01, ustar / kappa * (np.log((h - d) / z0m) - pm_h))
    uz0 = uh * np.exp(n * ((z0m + d) / h - 1))  # Eq. 32

    # Resistance components
    # Inertial sublayer (Eq. 41)
    if z > zr:
        rai = (1 / (kappa * ustar)) * (np.log((z - d) / (zr - d)) - ph_z + ph_zr)
    else:
        rai = 0.0

    # Roughness sublayer (Eq. 39)
    rar = (1 / (kappa * ustar)) * ((zr - h) / (zr - d)) - phs_zr + phs_h

    # Canopy layer (Eq. 42)
    exp_n = np.exp(n)
    exp_n_z0 = np.exp(n * (z0m + d) / h)
    rac = h * np.sinh(n) / (n * Kh) * (
        np.log((exp_n - 1) / (exp_n + 1)) -
        np.log((exp_n_z0 - 1) / (exp_n_z0 + 1))
    )

    # Within-canopy to soil (Eq. 43)
    exp_n_soil = np.exp(n * 0.01 / h)
    rws = h * np.sinh(n) / (n * Kh) * (
        np.log((exp_n_z0 - 1) / (exp_n_z0 + 1)) -
        np.log((exp_n_soil - 1) / (exp_n_soil + 1))
    )

    # Total resistances
    raa = rai + rar + rac  # Above canopy
    rawc = rwc  # Within canopy (leaves)
    raws = rws + rbs  # Within canopy (soil)

    return ResistanceOutput(
        ustar=ustar,
        raa=raa,
        rawc=rawc,
        raws=raws,
        rai=rai,
        rar=rar,
        rac=rac,
        rws=rws,
        uz0=uz0,
        Kh=Kh,
    )
