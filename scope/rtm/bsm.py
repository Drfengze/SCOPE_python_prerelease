"""Brightness-Shape-Moisture (BSM) soil reflectance model.

Translated from: src/RTMs/BSM.m

The BSM model simulates soil reflectance based on:
- Brightness parameter
- Shape parameters (lat/lon in spectral space)
- Soil moisture content

References:
    Verhoef, W., & Jia, L. (2014). GSV: A general model for hyperspectral
    soil reflectance simulation. International Journal of Applied Earth
    Observation and Geoinformation, 27, 17-25.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson


@dataclass
class BSMParameters:
    """Empirical parameters for the BSM soil moisture effect.

    Attributes:
        SMC: Soil moisture capacity parameter (recommended 0.25)
        film: Effective optical thickness of single water film (recommended 0.015)
    """

    SMC: float = 25.0  # Soil moisture capacity
    film: float = 0.015  # Water film optical thickness


@dataclass
class BSMSpectra:
    """Spectral inputs for BSM model.

    Attributes:
        GSV: Global Soil Vectors spectra, shape (nwl, 3)
        Kw: Water absorption coefficient spectrum, shape (nwl,)
        nw: Water refraction index spectrum, shape (nwl,)
    """

    GSV: NDArray[np.float64]  # (nwl, 3)
    Kw: NDArray[np.float64]  # (nwl,)
    nw: NDArray[np.float64]  # (nwl,)


def tav(alfa: float, nr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate average transmittance for isotropic incidence.

    This is the transmittance averaged over all incident angles,
    used for the Fresnel equations at the leaf-air interface.

    Based on Stern's formula in Lekner & Dorf (1988).

    Args:
        alfa: Incidence angle [degrees]
        nr: Refractive index array, shape (nwl,)

    Returns:
        Average transmittance Tav, shape (nwl,)
    """
    n2 = nr ** 2
    np_ = n2 + 1
    nm = n2 - 1

    a = ((nr + 1) ** 2) / 2
    k = -((n2 - 1) ** 2) / 4

    sin_a = np.sin(np.radians(alfa))

    if alfa != 0:
        B2 = sin_a ** 2 - np_ / 2
        B1 = np.sqrt(B2 ** 2 + k) if alfa != 90 else 0

        b = B1 - B2
        b3 = b ** 3
        a3 = a ** 3

        ts = (k ** 2 / (6 * b3) + k / b - b / 2) - (k ** 2 / (6 * a3) + k / a - a / 2)

        tp1 = -2 * n2 * (b - a) / (np_ ** 2)
        tp2 = -2 * n2 * np_ * np.log(b / a) / (nm ** 2)
        tp3 = n2 * (1 / b - 1 / a) / 2
        tp4 = 16 * n2 ** 2 * (n2 ** 2 + 1) * np.log((2 * np_ * b - nm ** 2) / (2 * np_ * a - nm ** 2)) / (np_ ** 3 * nm ** 2)
        tp5 = 16 * n2 ** 2 * n2 * (1 / (2 * np_ * b - nm ** 2) - 1 / (2 * np_ * a - nm ** 2)) / (np_ ** 3)

        tp = tp1 + tp2 + tp3 + tp4 + tp5
        Tav = (ts + tp) / (2 * sin_a ** 2)
    else:
        Tav = 4 * nr / ((nr + 1) ** 2)

    return Tav


def soilwat(
    rdry: NDArray[np.float64],
    nw: NDArray[np.float64],
    kw: NDArray[np.float64],
    SMp: float,
    SMC: float,
    deleff: float,
) -> NDArray[np.float64]:
    """Calculate wet soil reflectance from dry soil reflectance.

    Models the effect of water films on soil surface reflectance using
    a Poisson distribution of water film thicknesses.

    Args:
        rdry: Dry soil reflectance spectrum, shape (nwl,)
        nw: Water refraction index spectrum, shape (nwl,)
        kw: Water absorption coefficient spectrum, shape (nwl,)
        SMp: Soil moisture volume percentage [%]
        SMC: Soil moisture capacity parameter
        deleff: Effective optical thickness of single water film

    Returns:
        Wet soil reflectance spectrum, shape (nwl,)
    """
    # Mu-parameter of Poisson distribution
    mu = (SMp - 5) / SMC

    if mu <= 0:
        # Dry soil case
        return rdry.copy()

    # Lekner & Dorf (1988) modified soil background reflectance
    # for soil refraction index = 2.0
    rbac = 1 - (1 - rdry) * (rdry * tav(90, 2.0 / nw) / tav(90, np.array([2.0])) + 1 - rdry)

    # Total reflectance at bottom of water film surface
    # rho21: water to air, diffuse
    p = 1 - tav(90, nw) / nw ** 2

    # Reflectance of water film top surface (40 degrees incidence)
    # rho12: air to water, direct
    Rw = 1 - tav(40, nw)

    # Number of water films considered (0 to 6)
    k = np.arange(7)
    nk = len(k)

    # Poisson probability for each number of films
    fmul = poisson.pmf(k, mu)

    # Two-way transmittance through k water films
    # Shape: (nwl, nk) - broadcasting kw (nwl,) with k (nk,)
    tw = np.exp(-2 * np.outer(kw, deleff * k))

    # Wet reflectance for each number of films
    # For k=0 (dry), this reduces to rdry
    # Shape: (nwl, nk)
    rbac_col = rbac[:, np.newaxis]
    p_col = p[:, np.newaxis]
    Rw_col = Rw[:, np.newaxis]

    Rwet_k = Rw_col + (1 - Rw_col) * (1 - p_col) * tw * rbac_col / (1 - p_col * tw * rbac_col)

    # Weighted sum: dry fraction + wet fractions
    rwet = rdry * fmul[0] + np.dot(Rwet_k[:, 1:nk], fmul[1:nk])

    return rwet


def bsm(
    brightness: float,
    lat: float,
    lon: float,
    SMC_fraction: float,
    spectra: BSMSpectra,
    params: Optional[BSMParameters] = None,
) -> NDArray[np.float64]:
    """Calculate soil reflectance using the BSM model.

    The BSM model uses Global Soil Vectors (GSV) to simulate soil
    reflectance based on brightness and spectral shape parameters.

    Args:
        brightness: Soil brightness parameter [0-1]
        lat: Spectral shape latitude parameter [degrees, 20-40]
        lon: Spectral shape longitude parameter [degrees, 45-65]
        SMC_fraction: Soil moisture content as fraction [0-1]
        spectra: BSMSpectra containing GSV, Kw, and nw
        params: BSMParameters for moisture effect (optional)

    Returns:
        Soil reflectance spectrum, shape (nwl,)

    Example:
        >>> spectra = BSMSpectra(GSV=gsv_data, Kw=kw_data, nw=nw_data)
        >>> refl = bsm(0.5, 25.0, 45.0, 0.20, spectra)
    """
    if params is None:
        params = BSMParameters()

    GSV = spectra.GSV
    kw = spectra.Kw
    nw = spectra.nw

    # Convert SMC from fraction to percentage
    SMp = SMC_fraction * 100.0

    # Calculate dry soil reflectance from GSV
    # Using spherical coordinates to combine the three GSV vectors
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    f1 = brightness * np.sin(lat_rad)
    f2 = brightness * np.cos(lat_rad) * np.sin(lon_rad)
    f3 = brightness * np.cos(lat_rad) * np.cos(lon_rad)

    rdry = f1 * GSV[:, 0] + f2 * GSV[:, 1] + f3 * GSV[:, 2]

    # Apply soil moisture effect
    rwet = soilwat(rdry, nw, kw, SMp, params.SMC, params.film)

    return rwet


def bsm_from_soil(
    soil,
    spectra: BSMSpectra,
    params: Optional[BSMParameters] = None,
) -> NDArray[np.float64]:
    """Calculate soil reflectance from a Soil dataclass.

    Convenience function that extracts BSM parameters from a Soil object.

    Args:
        soil: Soil dataclass with BSMBrightness, BSMlat, BSMlon, SMC
        spectra: BSMSpectra containing GSV, Kw, and nw
        params: BSMParameters for moisture effect (optional)

    Returns:
        Soil reflectance spectrum, shape (nwl,)
    """
    return bsm(
        brightness=soil.BSMBrightness,
        lat=soil.BSMlat,
        lon=soil.BSMlon,
        SMC_fraction=soil.SMC,
        spectra=spectra,
        params=params,
    )
