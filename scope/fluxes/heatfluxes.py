"""Sensible and latent heat flux calculations.

Translated from: src/fluxes/heatfluxes.m

Calculates the sensible and latent heat fluxes from leaves and soil
based on temperature gradients and resistances.
"""

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..constants import CONSTANTS
from ..supporting.physics import satvap, slope_satvap


@dataclass
class HeatFluxOutput:
    """Output from heat flux calculations.

    Attributes:
        lE: Latent heat flux [W m-2]
        H: Sensible heat flux [W m-2]
        ec: Vapor pressure at leaf surface [hPa]
        Cc: CO2 concentration at leaf surface [µmol m-3]
        lambda_: Latent heat of vaporization [J kg-1]
        s: Slope of saturation vapor pressure curve [hPa K-1]
    """

    lE: Union[float, NDArray[np.float64]] = 0.0
    H: Union[float, NDArray[np.float64]] = 0.0
    ec: Union[float, NDArray[np.float64]] = 0.0
    Cc: Union[float, NDArray[np.float64]] = 0.0
    lambda_: Union[float, NDArray[np.float64]] = 0.0
    s: Union[float, NDArray[np.float64]] = 0.0


def latent_heat_vaporization(T: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """Calculate latent heat of vaporization.

    Args:
        T: Temperature [°C]

    Returns:
        Latent heat of vaporization [J kg-1]
    """
    return (2.501 - 0.002361 * T) * 1e6


def heatfluxes(
    ra: Union[float, NDArray[np.float64]],
    rs: Union[float, NDArray[np.float64]],
    Tc: Union[float, NDArray[np.float64]],
    ea: Union[float, NDArray[np.float64]],
    Ta: Union[float, NDArray[np.float64]],
    e_to_q: float,
    Ca: Union[float, NDArray[np.float64]],
    Ci: Union[float, NDArray[np.float64]],
    constants=None,
    es_fun=None,
    s_fun=None,
) -> dict:
    """Calculate latent and sensible heat fluxes.

    Translated from MATLAB heatfluxes.m

    Computes heat fluxes from a leaf or soil surface based on
    temperature gradients and resistance networks.

    Args:
        ra: Aerodynamic resistance for heat [s m-1]
        rs: Stomatal (or surface) resistance [s m-1]
        Tc: Surface temperature [°C]
        ea: Vapor pressure above canopy [hPa]
        Ta: Air temperature above canopy [°C]
        e_to_q: Conversion from vapor pressure to specific humidity [hPa-1]
        Ca: Ambient CO2 concentration [µmol m-3]
        Ci: Internal CO2 concentration [µmol m-3]
        constants: Physical constants (optional, uses default if None)
        es_fun: Saturation vapor pressure function es(T) (optional)
        s_fun: Slope of saturation curve function s(es, T) (optional)

    Returns:
        Dictionary with lE, H, ec, Cc, lambda_, s

    Example:
        >>> flux = heatfluxes(ra=50, rs=100, Tc=25, ea=15, Ta=20, e_to_q=0.001, Ca=400, Ci=250, constants=CONSTANTS)
        >>> print(f"Latent heat: {flux['lE']:.1f} W/m²")
    """
    # Use default constants if not provided
    if constants is None:
        rhoa = CONSTANTS.rhoa
        cp = CONSTANTS.cp
    else:
        rhoa = constants.rhoa
        cp = constants.cp

    # Use provided saturation functions or defaults
    if es_fun is None:
        ei = satvap(Tc)
    else:
        ei = es_fun(Tc)

    if s_fun is None:
        s = slope_satvap(Tc)
    else:
        s = s_fun(ei, Tc)

    # Latent heat of vaporization
    lambda_ = latent_heat_vaporization(Tc)

    # Specific humidity at surface and ambient
    qi = ei * e_to_q
    qa = ea * e_to_q

    # Latent heat flux [W m-2]
    lE = rhoa / (ra + rs) * lambda_ * (qi - qa)

    # Sensible heat flux [W m-2]
    H = (rhoa * cp) / ra * (Tc - Ta)

    # Vapor pressure at leaf surface [hPa]
    ec = ea + (ei - ea) * ra / (ra + rs)

    # CO2 concentration at leaf surface [µmol m-3]
    Cc = Ca - (Ca - Ci) * ra / (ra + rs)

    return {
        'lE': lE,
        'H': H,
        'ec': ec,
        'Cc': Cc,
        'lambda_': lambda_,
        's': s,
    }


def soil_heat_flux(
    Rn: Union[float, NDArray[np.float64]],
    method: int = 0,
    G_fraction: float = 0.35,
) -> Union[float, NDArray[np.float64]]:
    """Calculate soil heat flux.

    Args:
        Rn: Net radiation at soil surface [W m-2]
        method: Calculation method:
            0 = constant fraction of Rn
            1 = force-restore method (not implemented)
            2 = full soil heat transfer (not implemented)
        G_fraction: Fraction of Rn going to soil heat flux (for method 0)

    Returns:
        Soil heat flux [W m-2]
    """
    if method == 0:
        # Simple constant fraction
        return G_fraction * Rn
    else:
        # Force-restore and full models would require temperature history
        # For now, fall back to simple fraction
        return G_fraction * Rn


def penman_monteith(
    Rn: Union[float, NDArray[np.float64]],
    G: Union[float, NDArray[np.float64]],
    Ta: Union[float, NDArray[np.float64]],
    ea: Union[float, NDArray[np.float64]],
    ra: Union[float, NDArray[np.float64]],
    rs: Union[float, NDArray[np.float64]],
    rhoa: float = CONSTANTS.rhoa,
    cp: float = CONSTANTS.cp,
) -> Tuple[Union[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]]:
    """Calculate evapotranspiration using Penman-Monteith equation.

    Args:
        Rn: Net radiation [W m-2]
        G: Soil heat flux [W m-2]
        Ta: Air temperature [°C]
        ea: Actual vapor pressure [hPa]
        ra: Aerodynamic resistance [s m-1]
        rs: Surface (stomatal) resistance [s m-1]
        rhoa: Air density [kg m-3]
        cp: Specific heat of air [J kg-1 K-1]

    Returns:
        Tuple of (lE, H) latent and sensible heat fluxes [W m-2]
    """
    # Saturation vapor pressure and slope
    es = satvap(Ta)
    s = slope_satvap(Ta)

    # Vapor pressure deficit
    VPD = es - ea

    # Psychrometric constant (approx.)
    gamma = 0.665e-3 * 970 * 100  # hPa/K at sea level

    # Latent heat of vaporization
    lambda_ = latent_heat_vaporization(Ta)

    # Available energy
    A = Rn - G

    # Penman-Monteith equation
    numerator = s * A + rhoa * cp * VPD * 100 / ra  # VPD in Pa
    denominator = s + gamma * (1 + rs / ra)

    lE = numerator / denominator
    H = A - lE

    return lE, H
