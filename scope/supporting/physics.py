"""Physical utility functions for SCOPE model.

Translated from: src/supporting/ephoton.m, satvap.m, Planck.m
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..constants import CONSTANTS


def ephoton(
    wavelength: Union[float, NDArray[np.float64]],
    h: float = CONSTANTS.h,
    c: float = CONSTANTS.c,
) -> Union[float, NDArray[np.float64]]:
    """Calculate energy of a single photon at given wavelength.

    Args:
        wavelength: Wavelength in meters [m]
        h: Planck's constant [J s] (default: from constants)
        c: Speed of light [m s-1] (default: from constants)

    Returns:
        Energy per photon [J]

    Example:
        >>> # Energy of a photon at 550 nm (green light)
        >>> E = ephoton(550e-9)
    """
    return h * c / wavelength


def e2phot(
    wavelength: Union[float, NDArray[np.float64]],
    energy: Union[float, NDArray[np.float64]],
    h: float = CONSTANTS.h,
    c: float = CONSTANTS.c,
) -> Union[float, NDArray[np.float64]]:
    """Convert energy to number of photons at given wavelength.

    Args:
        wavelength: Wavelength in meters [m]
        energy: Energy [J] or energy flux [W m-2]
        h: Planck's constant [J s]
        c: Speed of light [m s-1]

    Returns:
        Number of photons or photon flux [photons or photons m-2 s-1]
    """
    return energy * wavelength / (h * c)


def phot2e(
    wavelength: Union[float, NDArray[np.float64]],
    photons: Union[float, NDArray[np.float64]],
    h: float = CONSTANTS.h,
    c: float = CONSTANTS.c,
) -> Union[float, NDArray[np.float64]]:
    """Convert number of photons to energy at given wavelength.

    Args:
        wavelength: Wavelength in meters [m]
        photons: Number of photons or photon flux
        h: Planck's constant [J s]
        c: Speed of light [m s-1]

    Returns:
        Energy [J] or energy flux [W m-2]
    """
    return photons * h * c / wavelength


def satvap(T: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """Calculate saturated vapor pressure at temperature T.

    Uses the Magnus-Tetens approximation:
        es(T) = 6.107 * 10^(7.5*T / (237.3 + T))

    Args:
        T: Temperature in degrees Celsius [°C]

    Returns:
        Saturated vapor pressure [hPa or mbar]

    Example:
        >>> es = satvap(20.0)  # ~23.4 hPa at 20°C
    """
    a = 7.5
    b = 237.3  # degrees C
    return 6.107 * 10.0 ** (a * T / (b + T))


def slope_satvap(T: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """Calculate slope of saturation vapor pressure curve.

    The slope s = d(es)/dT is used in Penman-Monteith equation.

    Args:
        T: Temperature in degrees Celsius [°C]

    Returns:
        Slope of saturation curve [hPa K-1]

    Example:
        >>> s = slope_satvap(20.0)  # ~1.45 hPa/K at 20°C
    """
    a = 7.5
    b = 237.3  # degrees C
    es = satvap(T)
    return es * 2.3026 * a * b / (b + T) ** 2  # 2.3026 = ln(10), matches MATLAB


def planck(
    wl: Union[float, NDArray[np.float64]],
    T: Union[float, NDArray[np.float64]],
    emissivity: Union[float, NDArray[np.float64]] = 1.0,
) -> Union[float, NDArray[np.float64]]:
    """Calculate Planck blackbody radiance.

    Computes spectral radiance according to Planck's law:
        L = em * c1 * wl^(-5) / (exp(c2/(wl*T)) - 1)

    where c1 and c2 are radiation constants.

    Args:
        wl: Wavelength in nanometers [nm]
        T: Temperature in Kelvin [K]
        emissivity: Emissivity [-] (default: 1.0 for blackbody)

    Returns:
        Spectral radiance [W m-2 sr-1 nm-1]

    Note:
        For arrays, wl and T can be broadcast together.
        Common usage: wl is (nwl,) and T is scalar, or
                      wl is (nwl,1) and T is (nlayers,) for 2D output.
    """
    # First radiation constant: 2*h*c^2 = 1.191066e-22 W m^4 sr^-1
    c1 = 1.191066e-22
    # Second radiation constant: h*c/k = 14388.33 µm K
    c2 = 14388.33

    # Convert wavelength from nm to m for c1 term, nm to µm for c2 term
    wl_m = wl * 1e-9      # nm to m
    wl_um = wl * 1e-3     # nm to µm

    # Planck function
    Lb = emissivity * c1 * wl_m ** (-5) / (np.exp(c2 / (wl_um * T)) - 1)

    return Lb


def stefan_boltzmann(
    T: Union[float, NDArray[np.float64]],
    emissivity: float = 1.0,
    sigma: float = CONSTANTS.sigmaSB,
) -> Union[float, NDArray[np.float64]]:
    """Calculate total thermal radiance using Stefan-Boltzmann law.

    Args:
        T: Temperature in Kelvin [K]
        emissivity: Surface emissivity [-] (default: 1.0)
        sigma: Stefan-Boltzmann constant [W m-2 K-4]

    Returns:
        Total radiant emittance [W m-2]

    Example:
        >>> L = stefan_boltzmann(300)  # ~459 W/m² at 300K
    """
    return emissivity * sigma * T ** 4


def vapor_pressure_deficit(
    T: Union[float, NDArray[np.float64]],
    ea: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate vapor pressure deficit.

    VPD = es(T) - ea, where es is saturation vapor pressure.

    Args:
        T: Air temperature [°C]
        ea: Actual vapor pressure [hPa]

    Returns:
        Vapor pressure deficit [hPa]
    """
    return satvap(T) - ea


def relative_humidity(
    T: Union[float, NDArray[np.float64]],
    ea: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Calculate relative humidity.

    RH = ea / es(T) * 100

    Args:
        T: Air temperature [°C]
        ea: Actual vapor pressure [hPa]

    Returns:
        Relative humidity [%]
    """
    return ea / satvap(T) * 100.0
