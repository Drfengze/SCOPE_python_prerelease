"""Optical Radiative Transfer Model (RTMo) for canopy.

Translated from: src/RTMs/RTMo.m

RTMo calculates hemispherical and directional visible and thermal radiation,
as well as gap probabilities for a vegetation canopy.

References:
    Verhoef, W. (1998). Theory of radiative transfer models applied in
    optical remote sensing of vegetation canopies. PhD Thesis.

    Verhoef, W., Jia, L., Xiao, Q., & Su, Z. (2007). Unified optical-thermal
    four-stream radiative transfer theory for homogeneous vegetation canopies.
    IEEE Transactions on Geoscience and Remote Sensing, 45(6), 1808-1822.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from numba import jit
from scipy import integrate

from ..constants import CONSTANTS
from ..supporting.integration import sint
from ..supporting.physics import e2phot, planck

# Constants
AVOGADRO = 6.02214076e23
h = 6.6262e-34  # Planck's constant
c = 299792458.0  # Speed of light


# ============================================================================
# Numba-optimized helper functions
# ============================================================================
@jit(nopython=True, cache=True)
def sint_1d(y, x):
    """Fast trapezoidal integration for 1D arrays."""
    n = len(x)
    result = 0.0
    for i in range(n - 1):
        result += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return result


@jit(nopython=True, cache=True)
def e2phot_numba(wavelength_m, E):
    """Convert energy to photons. wavelength in meters."""
    n = len(wavelength_m)
    result = np.empty(n)
    for i in range(n):
        e_photon = h * c / wavelength_m[i]
        result[i] = E[i] / e_photon / AVOGADRO
    return result


@jit(nopython=True, cache=True)
def pso_integrand(K, k, LAI, q, dso, y):
    """Pso integrand function."""
    if dso != 0:
        alf = (dso / q) * 2 / (k + K)
        return np.exp((K + k) * LAI * y + np.sqrt(K * k) * LAI / alf * (1 - np.exp(y * alf)))
    else:
        return np.exp((K + k) * LAI * y - np.sqrt(K * k) * LAI * y)


@jit(nopython=True, cache=True)
def pso_integral(K, k, LAI, q, dso, xl_lower, xl_upper, n_points=10):
    """Numerical integration of Pso using Simpson's rule."""
    h = (xl_upper - xl_lower) / n_points
    result = pso_integrand(K, k, LAI, q, dso, xl_lower) + pso_integrand(K, k, LAI, q, dso, xl_upper)

    for i in range(1, n_points):
        x = xl_lower + i * h
        if i % 2 == 0:
            result += 2 * pso_integrand(K, k, LAI, q, dso, x)
        else:
            result += 4 * pso_integrand(K, k, LAI, q, dso, x)

    return result * h / 3


@jit(nopython=True, cache=True)
def calc_Pso_array(K, k, LAI, q, dso, xl, dx, nl):
    """Calculate Pso for all layers."""
    Pso = np.zeros(nl + 1)
    for j in range(nl + 1):
        xl_j = xl[j]
        xl_lower = xl_j - dx
        Pso[j] = pso_integral(K, k, LAI, q, dso, xl_lower, xl_j) / dx
    return Pso


@jit(nopython=True, cache=True)
def calc_layer_absorption_direct(
    nl, nwlP, nwlPAR,
    Esun_IwlP, Esun_Ipar, Esun_full,
    epsc_2d, kChlrel_2d, kCarrel_2d,
    wl_full, wlP, wlPAR, wlP_m, wlPAR_m,
):
    """Calculate absorbed radiation per layer from direct solar.

    All arrays must be pre-sliced to correct wavelength ranges.
    """
    Asun = np.zeros(nl)
    Pnsun = np.zeros(nl)
    Rnsun_PAR = np.zeros(nl)
    Pnsun_Cab = np.zeros(nl)
    Rnsun_Cab = np.zeros(nl)
    Pnsun_Car = np.zeros(nl)
    Rnsun_Car = np.zeros(nl)

    for j in range(nl):
        epsc_j = epsc_2d[j]
        kChlrel_j = kChlrel_2d[j]
        kCarrel_j = kCarrel_2d[j]

        # Full spectrum integral
        Asun[j] = 0.001 * sint_1d(Esun_full * epsc_j, wl_full)

        # Cab/Car absorption (optical range)
        Rnsun_Cab[j] = 0.001 * sint_1d(Esun_IwlP * epsc_j[:nwlP] * kChlrel_j, wlP)
        Rnsun_Car[j] = 0.001 * sint_1d(Esun_IwlP * epsc_j[:nwlP] * kCarrel_j, wlP)

        # PAR
        Rnsun_PAR[j] = 0.001 * sint_1d(Esun_Ipar * epsc_j[:nwlPAR], wlPAR)

        # Photon fluxes
        Pnsun[j] = 0.001 * sint_1d(e2phot_numba(wlPAR_m, Esun_Ipar * epsc_j[:nwlPAR]), wlPAR)
        Pnsun_Cab[j] = 0.001 * sint_1d(e2phot_numba(wlP_m, kChlrel_j * Esun_IwlP * epsc_j[:nwlP]), wlP)
        Pnsun_Car[j] = 0.001 * sint_1d(e2phot_numba(wlP_m, kCarrel_j * Esun_IwlP * epsc_j[:nwlP]), wlP)

    return Asun, Pnsun, Rnsun_PAR, Pnsun_Cab, Rnsun_Cab, Pnsun_Car, Rnsun_Car


@jit(nopython=True, cache=True)
def calc_layer_absorption_diffuse(
    nl, nwl, nwlP, nwlPAR,
    Emin_, Eplu_,
    epsc_2d, kChlrel_2d, kCarrel_2d,
    wl_full, wlP, wlPAR, wlP_m, wlPAR_m,
):
    """Calculate absorbed radiation per layer from diffuse radiation.

    Note: Assumes wavelengths are ordered with PAR/optical at the start.
    kChlrel_2d and kCarrel_2d should be pre-sliced to optical range (nwlP).
    """
    Rndif = np.zeros(nl)
    Pndif = np.zeros(nl)
    Pndif_Cab = np.zeros(nl)
    Rndif_Cab = np.zeros(nl)
    Pndif_Car = np.zeros(nl)
    Rndif_Car = np.zeros(nl)
    Rndif_PAR = np.zeros(nl)

    for j in range(nl):
        # Diffuse incident radiation
        E_ = np.zeros(nwl)
        for i in range(nwl):
            E_[i] = 0.5 * (Emin_[j, i] + Emin_[j + 1, i] + Eplu_[j, i] + Eplu_[j + 1, i])

        epsc_j = epsc_2d[j]
        kChlrel_j = kChlrel_2d[j]
        kCarrel_j = kCarrel_2d[j]

        # Net radiation (full spectrum)
        Rndif_j = E_ * epsc_j
        Rndif[j] = 0.001 * sint_1d(Rndif_j, wl_full)

        # PAR energy
        Rndif_PAR[j] = 0.001 * sint_1d(Rndif_j[:nwlPAR], wlPAR)

        # Photon flux
        Pndif[j] = 0.001 * sint_1d(e2phot_numba(wlPAR_m, Rndif_j[:nwlPAR]), wlPAR)

        # Cab absorption
        Rndif_j_P = Rndif_j[:nwlP]
        Pndif_Cab[j] = 0.001 * sint_1d(e2phot_numba(wlP_m, kChlrel_j * Rndif_j_P), wlP)
        Rndif_Cab[j] = 0.001 * sint_1d(kChlrel_j * Rndif_j_P, wlP)

        # Car absorption
        Pndif_Car[j] = 0.001 * sint_1d(e2phot_numba(wlP_m, kCarrel_j * Rndif_j_P), wlP)
        Rndif_Car[j] = 0.001 * sint_1d(kCarrel_j * Rndif_j_P, wlP)

    return Rndif, Pndif, Pndif_Cab, Rndif_Cab, Pndif_Car, Rndif_Car, Rndif_PAR


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class GapProbabilities:
    """Gap fraction probabilities.

    Attributes:
        k: Extinction coefficient in solar direction
        K: Extinction coefficient in viewing direction
        Ps: Gap probability in solar direction, shape (nl+1,)
        Po: Gap probability in viewing direction, shape (nl+1,)
        Pso: Bidirectional gap probability, shape (nl+1,)
    """

    k: float = 0.0
    K: float = 0.0
    Ps: NDArray[np.float64] = None
    Po: NDArray[np.float64] = None
    Pso: NDArray[np.float64] = None


@dataclass
class RadiativeTransferOutput:
    """Output from RTMo canopy optical radiative transfer.

    Contains spectral fluxes, reflectances, and absorbed radiation.
    """

    # Canopy reflectance factors
    rsd: NDArray[np.float64] = None  # TOC directional-hemispherical reflectance
    rdd: NDArray[np.float64] = None  # TOC hemispherical-hemispherical reflectance
    rdo: NDArray[np.float64] = None  # TOC hemispherical-directional reflectance
    rso: NDArray[np.float64] = None  # TOC directional-directional reflectance
    refl: NDArray[np.float64] = None  # TOC reflectance

    # Layer optical properties
    rho_dd: NDArray[np.float64] = None  # diffuse-diffuse reflectance per layer
    tau_dd: NDArray[np.float64] = None  # diffuse-diffuse transmittance per layer
    rho_sd: NDArray[np.float64] = None  # direct-diffuse reflectance per layer
    tau_ss: NDArray[np.float64] = None  # direct-direct transmittance per layer
    tau_sd: NDArray[np.float64] = None  # direct-diffuse transmittance per layer
    R_sd: NDArray[np.float64] = None  # cumulative direct-diffuse reflectance
    R_dd: NDArray[np.float64] = None  # cumulative diffuse-diffuse reflectance

    # Scattering coefficients
    vb: NDArray[np.float64] = None  # directional backscatter coefficient
    vf: NDArray[np.float64] = None  # directional forward scatter coefficient
    sigf: NDArray[np.float64] = None  # forward scatter coefficient
    sigb: NDArray[np.float64] = None  # backscatter coefficient

    # Transmittance factors
    Xdd: NDArray[np.float64] = None
    Xsd: NDArray[np.float64] = None
    Xss: NDArray[np.float64] = None

    # Spectral fluxes
    Esun_: NDArray[np.float64] = None  # Incident solar spectrum (nwl,)
    Esky_: NDArray[np.float64] = None  # Incident sky spectrum (nwl,)
    Emin_: NDArray[np.float64] = None  # Downward diffuse in canopy (nl+1, nwl)
    Eplu_: NDArray[np.float64] = None  # Upward diffuse in canopy (nl+1, nwl)
    Emins_: NDArray[np.float64] = None  # Downward diffuse from direct solar (nl+1, nwl)
    Emind_: NDArray[np.float64] = None  # Downward diffuse from sky (nl+1, nwl)
    Eplus_: NDArray[np.float64] = None  # Upward diffuse from direct solar (nl+1, nwl)
    Eplud_: NDArray[np.float64] = None  # Upward diffuse from sky (nl+1, nwl)
    Lo_: NDArray[np.float64] = None  # TOC radiance in observation dir (nwl,)
    Eout_: NDArray[np.float64] = None  # TOC upward radiation (nwl,)

    # Integrated values
    PAR: float = 0.0  # Incident PAR (umol m-2 s-1)
    EPAR: float = 0.0  # Incident PAR energy (W m-2)
    Eouto: float = 0.0  # Upward optical radiation (W m-2)
    Eoutt: float = 0.0  # Upward thermal radiation (W m-2)
    Lot: float = 0.0  # Thermal radiance in observation direction (W m-2)

    # Net radiation per layer - SHADED leaves (diffuse only)
    Rnhc: NDArray[np.float64] = None  # Net radiation shaded leaves (nl,) W/m2
    Pnh: NDArray[np.float64] = None  # Net PAR shaded leaves (nl,) umol/m2/s
    Pnh_Cab: NDArray[np.float64] = None  # PAR absorbed by Chl, shaded (nl,) umol/m2/s
    Rnh_Cab: NDArray[np.float64] = None  # PAR absorbed by Chl, shaded (nl,) W/m2
    Pnh_Car: NDArray[np.float64] = None  # PAR absorbed by Car, shaded (nl,) umol/m2/s
    Rnh_Car: NDArray[np.float64] = None  # PAR absorbed by Car, shaded (nl,) W/m2
    Rnh_PAR: NDArray[np.float64] = None  # PAR energy, shaded (nl,) W/m2

    # Net radiation per layer - SUNLIT leaves (direct + diffuse)
    # Full mode: shape (nli, nlazi, nl) = (13, 36, nl)
    # Lite mode: shape (nl,)
    Rnuc: NDArray[np.float64] = None  # Net radiation sunlit leaves
    Pnu: NDArray[np.float64] = None  # Net PAR sunlit leaves umol/m2/s
    Pnu_Cab: NDArray[np.float64] = None  # PAR absorbed by Chl, sunlit
    Rnu_Cab: NDArray[np.float64] = None  # PAR absorbed by Chl, sunlit W/m2
    Pnu_Car: NDArray[np.float64] = None  # PAR absorbed by Car, sunlit
    Rnu_Car: NDArray[np.float64] = None  # PAR absorbed by Car, sunlit W/m2
    Rnu_PAR: NDArray[np.float64] = None  # PAR energy, sunlit W/m2

    # Soil radiation
    Rnhs: float = 0.0  # Net radiation shaded soil (W m-2)
    Rnus: float = 0.0  # Net radiation sunlit soil (W m-2)


def volscat(
    tts: float,
    tto: float,
    psi: float,
    ttli: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate volume scattering functions.

    Computes the extinction coefficients and scattering phase functions
    for each leaf inclination angle class.

    Args:
        tts: Solar zenith angle [degrees]
        tto: Observer zenith angle [degrees]
        psi: Relative azimuth angle [degrees]
        ttli: Leaf inclination angles [degrees], shape (nli,)

    Returns:
        Tuple of (chi_s, chi_o, frho, ftau):
        - chi_s: Ross-Nilson function, solar direction (nli,)
        - chi_o: Ross-Nilson function, observer direction (nli,)
        - frho: Backward scattering weight (nli,)
        - ftau: Forward scattering weight (nli,)
    """
    deg2rad = np.pi / 180
    nli = len(ttli)

    psi_rad = psi * deg2rad * np.ones(nli)
    cos_psi = np.cos(psi * deg2rad)

    cos_ttli = np.cos(ttli * deg2rad)
    sin_ttli = np.sin(ttli * deg2rad)

    cos_tts = np.cos(tts * deg2rad)
    sin_tts = np.sin(tts * deg2rad)

    cos_tto = np.cos(tto * deg2rad)
    sin_tto = np.sin(tto * deg2rad)

    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts

    Co = cos_ttli * cos_tto
    So = sin_ttli * sin_tto

    As = np.maximum(Ss, Cs)
    Ao = np.maximum(So, Co)

    bts = np.arccos(np.clip(-Cs / As, -1, 1))
    bto = np.arccos(np.clip(-Co / Ao, -1, 1))

    chi_o = 2 / np.pi * ((bto - np.pi / 2) * Co + np.sin(bto) * So)
    chi_s = 2 / np.pi * ((bts - np.pi / 2) * Cs + np.sin(bts) * Ss)

    delta1 = np.abs(bts - bto)
    delta2 = np.pi - np.abs(bts + bto - np.pi)

    Tot = psi_rad + delta1 + delta2

    bt1 = np.minimum(psi_rad, delta1)
    bt3 = np.maximum(psi_rad, delta2)
    bt2 = Tot - bt1 - bt3

    T1 = 2 * Cs * Co + Ss * So * cos_psi
    T2 = np.sin(bt2) * (2 * As * Ao + Ss * So * np.cos(bt1) * np.cos(bt3))

    Jmin = bt2 * T1 - T2
    Jplus = (np.pi - bt2) * T1 + T2

    frho = np.maximum(0, Jplus / (2 * np.pi ** 2))
    ftau = np.maximum(0, -Jmin / (2 * np.pi ** 2))

    return chi_s, chi_o, frho, ftau


def pso_function(K: float, k: float, LAI: float, q: float, dso: float, xl: float) -> float:
    """Calculate bidirectional gap probability at level xl.

    Args:
        K: Extinction coefficient, observer direction
        k: Extinction coefficient, solar direction
        LAI: Leaf area index
        q: Hotspot parameter
        dso: Angular distance sun-observer
        xl: Cumulative LAI level (0 at top, -1 at bottom)

    Returns:
        Pso value at level xl
    """
    if dso != 0:
        alf = (dso / q) * 2 / (k + K)
        pso = np.exp((K + k) * LAI * xl + np.sqrt(K * k) * LAI / alf * (1 - np.exp(xl * alf)))
    else:
        pso = np.exp((K + k) * LAI * xl - np.sqrt(K * k) * LAI * xl)

    return pso


def calc_reflectances(
    tau_ss: NDArray[np.float64],
    tau_sd: NDArray[np.float64],
    tau_dd: NDArray[np.float64],
    rho_dd: NDArray[np.float64],
    rho_sd: NDArray[np.float64],
    rs: NDArray[np.float64],
    nl: int,
    nwl: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate canopy reflectances by layer recursion.

    Args:
        tau_ss: Direct-direct transmittance (nl,)
        tau_sd: Direct-diffuse transmittance (nl, nwl)
        tau_dd: Diffuse-diffuse transmittance (nl, nwl)
        rho_dd: Diffuse-diffuse reflectance (nl, nwl)
        rho_sd: Direct-diffuse reflectance (nl, nwl)
        rs: Soil reflectance (nwl,)
        nl: Number of layers
        nwl: Number of wavelengths

    Returns:
        Tuple of (R_sd, R_dd, Xss, Xsd, Xdd) reflectance/transmittance arrays
    """
    R_sd = np.zeros((nl + 1, nwl))
    R_dd = np.zeros((nl + 1, nwl))
    Xsd = np.zeros((nl, nwl))
    Xdd = np.zeros((nl, nwl))
    Xss = np.zeros(nl)

    R_sd[nl, :] = rs
    R_dd[nl, :] = rs

    # From bottom to top
    for j in range(nl - 1, -1, -1):
        Xss[j] = tau_ss[j]
        dnorm = 1 - rho_dd[j, :] * R_dd[j + 1, :]
        Xsd[j, :] = (tau_sd[j, :] + tau_ss[j] * R_sd[j + 1, :] * rho_dd[j, :]) / dnorm
        Xdd[j, :] = tau_dd[j, :] / dnorm
        R_sd[j, :] = rho_sd[j, :] + tau_dd[j, :] * (R_sd[j + 1, :] * Xss[j] + R_dd[j + 1, :] * Xsd[j, :])
        R_dd[j, :] = rho_dd[j, :] + tau_dd[j, :] * R_dd[j + 1, :] * Xdd[j, :]

    return R_sd, R_dd, Xss, Xsd, Xdd


def calc_fluxprofile(
    Esun_: NDArray[np.float64],
    Esky_: NDArray[np.float64],
    rs: NDArray[np.float64],
    Xss: NDArray[np.float64],
    Xsd: NDArray[np.float64],
    Xdd: NDArray[np.float64],
    R_sd: NDArray[np.float64],
    R_dd: NDArray[np.float64],
    nl: int,
    nwl: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Calculate flux profiles through the canopy.

    Args:
        Esun_: Direct solar irradiance (nwl,)
        Esky_: Diffuse sky irradiance (nwl,)
        rs: Soil reflectance (nwl,)
        Xss, Xsd, Xdd: Transmittance factors
        R_sd, R_dd: Reflectance factors
        nl: Number of layers
        nwl: Number of wavelengths

    Returns:
        Tuple of (Emin_, Eplu_, Es_) downward diffuse, upward diffuse, and direct fluxes
    """
    Es_ = np.zeros((nl + 1, nwl))
    Emin_ = np.zeros((nl + 1, nwl))
    Eplu_ = np.zeros((nl + 1, nwl))

    Es_[0, :] = Esun_
    Emin_[0, :] = Esky_

    # From top to bottom
    for j in range(nl):
        Es_[j + 1, :] = Xss[j] * Es_[j, :]
        Emin_[j + 1, :] = Xsd[j, :] * Es_[j, :] + Xdd[j, :] * Emin_[j, :]
        Eplu_[j, :] = R_sd[j, :] * Es_[j, :] + R_dd[j, :] * Emin_[j, :]

    # Bottom boundary
    Eplu_[nl, :] = rs * (Es_[nl, :] + Emin_[nl, :])

    return Emin_, Eplu_, Es_


def calcTOCirr(
    atmo: dict,
    meteo,
    rdd: NDArray[np.float64],
    rsd: NDArray[np.float64],
    wl: NDArray[np.float64],
    nwl: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate top-of-canopy irradiance.

    Translated from MATLAB calcTOCirr function in RTMo.m

    Args:
        atmo: Atmospheric data dictionary
        meteo: Meteorological data
        rdd: Hemispherical-hemispherical reflectance
        rsd: Directional-hemispherical reflectance
        wl: Wavelengths (nm)
        nwl: Number of wavelengths

    Returns:
        Tuple of (Esun_, Esky_) incident solar and sky radiation
    """
    Fd = np.zeros(nwl)
    Ta_K = meteo.Ta + 273.15
    Ls = planck(wl, Ta_K)

    if atmo is not None and "Esun_" in atmo:
        # Use provided atmospheric spectra
        Esun_ = atmo["Esun_"]
        Esky_ = atmo["Esky_"]
    elif atmo is not None and "M" in atmo:
        # Calculate from MODTRAN parameters
        M = atmo["M"]
        t1 = M[:, 0]
        t3 = M[:, 1]
        t4 = M[:, 2]
        t5 = M[:, 3]
        t12 = M[:, 4]
        t16 = M[:, 5]

        Esun_ = np.maximum(1e-6, np.pi * t1 * t4)
        Esky_ = np.maximum(1e-6, np.pi / (1 - t3 * rdd) * (t1 * (t5 + t12 * rsd) + Fd + (1 - rdd) * Ls * t3 + t16))

        # Adjust for measured Rin if provided
        if hasattr(meteo, 'Rin') and meteo.Rin != -999:
            J_o = wl < 3000  # optical spectrum
            Esunto = 0.001 * sint(Esun_[J_o], wl[J_o])
            Eskyto = 0.001 * sint(Esky_[J_o], wl[J_o])
            Etoto = Esunto + Eskyto

            fEsuno = np.zeros(nwl)
            fEskyo = np.zeros(nwl)
            fEsuno[J_o] = Esun_[J_o] / Etoto
            fEskyo[J_o] = Esky_[J_o] / Etoto

            J_t = wl >= 3000  # thermal spectrum
            Esuntt = 0.001 * sint(Esun_[J_t], wl[J_t])
            Eskytt = 0.001 * sint(Esky_[J_t], wl[J_t])
            Etott = Eskytt + Esuntt

            fEsunt = np.zeros(nwl)
            fEskyt = np.zeros(nwl)
            fEsunt[J_t] = Esun_[J_t] / Etott
            fEskyt[J_t] = Esky_[J_t] / Etott

            Esun_[J_o] = fEsuno[J_o] * meteo.Rin
            Esky_[J_o] = fEskyo[J_o] * meteo.Rin
            Esun_[J_t] = fEsunt[J_t] * meteo.Rli
            Esky_[J_t] = fEskyt[J_t] * meteo.Rli
    else:
        # MATLAB always requires atmospheric data (MODTRAN or Esun_/Esky_).
        # For consistency with MATLAB, raise an error instead of using a fallback.
        raise ValueError(
            "Atmospheric data required. Provide either:\n"
            "  - atmo['M']: MODTRAN matrix (load from .atm file), or\n"
            "  - atmo['Esun_'] and atmo['Esky_']: spectral irradiance arrays\n"
            "MATLAB does not have a simple model fallback, so Python requires "
            "atmospheric data for consistency."
        )
        # # Simple model fallback (commented out for MATLAB consistency):
        # # If you need to use this, uncomment the code below. It properly handles
        # # both Rin (optical) and Rli (thermal) radiation.
        # #
        # # Initialize arrays
        # Esun_ = np.full(nwl, 1e-10)
        # Esky_ = np.full(nwl, 1e-10)
        # #
        # # Optical wavelengths (< 3000nm): use Rin
        # J_o = wl < 3000
        # if np.any(J_o) and meteo.Rin > 0:
        #     # Simple Gaussian-like solar spectrum shape (peak ~500nm)
        #     spectrum_o = np.exp(-0.5 * ((wl[J_o] - 500) / 300)**2)
        #     spectrum_o_integral = sint(spectrum_o, wl[J_o])
        #     if spectrum_o_integral > 0:
        #         # Split 70% direct, 30% diffuse; multiply by 1000 for mW/m²/nm
        #         Esun_[J_o] = 0.7 * meteo.Rin * 1000 / spectrum_o_integral * spectrum_o
        #         Esky_[J_o] = 0.3 * meteo.Rin * 1000 / spectrum_o_integral * spectrum_o
        # #
        # # Thermal wavelengths (>= 3000nm): use Rli
        # J_t = wl >= 3000
        # if np.any(J_t) and meteo.Rli > 0:
        #     # Thermal spectrum shape (peak ~10000nm for ~290K blackbody)
        #     spectrum_t = np.exp(-0.5 * ((wl[J_t] - 10000) / 5000)**2)
        #     spectrum_t_integral = sint(spectrum_t, wl[J_t])
        #     if spectrum_t_integral > 0:
        #         # All thermal is diffuse (from sky); multiply by 1000 for mW/m²/nm
        #         Esun_[J_t] = 1e-10  # No direct thermal from sun
        #         Esky_[J_t] = meteo.Rli * 1000 / spectrum_t_integral * spectrum_t

    return Esun_, Esky_


def rtmo(
    spectral,
    soil,
    leafopt,
    canopy,
    angles,
    meteo,
    atmo: Optional[dict] = None,
    options: Optional[dict] = None,
) -> Tuple[RadiativeTransferOutput, GapProbabilities, dict]:
    """Run optical radiative transfer model.

    Calculates spectrally resolved radiation fluxes through a vegetation
    canopy and at the top of canopy (TOC).

    Args:
        spectral: SpectralBands with wavelength definitions
        soil: Soil dataclass with reflectance
        leafopt: LeafOpticalOutput with refl, tran, kChlrel
        canopy: Canopy dataclass with LAI, lidf, etc.
        angles: Angles dataclass with tts, tto, psi
        meteo: Meteo dataclass with Rin, Rli, Ta
        atmo: Optional atmospheric data dictionary
        options: Optional simulation options dictionary with 'lite' flag

    Returns:
        Tuple of (RadiativeTransferOutput, GapProbabilities, profiles dict)
    """
    # Extract options
    lite = options.get("lite", False) if options else False
    calc_vert_profiles = options.get("calc_vert_profiles", True) if options else True

    # Extract parameters
    deg2rad = CONSTANTS.deg2rad
    wl = spectral.wlS
    nwl = len(wl)
    wlPAR = spectral.wlPAR
    Ipar = spectral.IwlPAR
    IwlP = spectral.IwlP
    wlP = spectral.wlP
    nwlP = len(wlP)  # Number of optical wavelengths (2001)
    IwlT = spectral.IwlT
    wlT = spectral.wlT

    tts = angles.tts
    tto = angles.tto
    psi = angles.psi

    nl = canopy.nlayers
    nli = canopy.nlincl
    litab = canopy.litab
    lazitab = canopy.lazitab
    nlazi = canopy.nlazi
    LAI = canopy.LAI
    lidf = canopy.lidf
    xl = canopy.xl
    dx = 1.0 / nl

    rho = leafopt.refl
    tau = leafopt.tran
    kChlrel = leafopt.kChlrel
    kCarrel = getattr(leafopt, 'kCarrel', np.ones_like(kChlrel))
    # Use full spectrum (2162 bands) for soil reflectance to match MATLAB
    rs = soil.refl if soil.refl is not None else np.full(nwl, 0.1)

    epsc = 1 - rho - tau  # Leaf emissivity
    epss = 1 - rs  # Soil emissivity
    iLAI = LAI / nl  # LAI per layer

    # 1. Geometric quantities
    # 1.1 general geometric quantities
    cos_tts = np.cos(tts * deg2rad)
    tan_tto = np.tan(tto * deg2rad)
    cos_tto = np.cos(tto * deg2rad)
    sin_tts = np.sin(tts * deg2rad)
    tan_tts = np.tan(tts * deg2rad)

    psi = abs(psi - 360 * round(psi / 360))
    dso = np.sqrt(tan_tts ** 2 + tan_tto ** 2 - 2 * tan_tts * tan_tto * np.cos(psi * deg2rad))

    # 1.2 geometric factors associated with extinction and scattering
    chi_s, chi_o, frho, ftau = volscat(tts, tto, psi, litab)

    cos_ttlo = np.cos(lazitab * deg2rad)  # [1, 36] cos leaf azimuth angles
    cos_ttli = np.cos(litab * deg2rad)  # [13] cos leaf angles
    sin_ttli = np.sin(litab * deg2rad)  # [13] sin leaf angles

    ksli = chi_s / cos_tts  # [13] extinction coefficient per leaf angle
    koli = chi_o / cos_tto  # [13] extinction coefficient per leaf angle

    sobli = frho * np.pi / (cos_tts * cos_tto)  # [13]
    sofli = ftau * np.pi / (cos_tts * cos_tto)  # [13]
    bfli = cos_ttli ** 2  # [13]

    # integration over angles -> scalars
    k = np.dot(ksli, lidf)  # extinction coefficient in direction of sun
    K = np.dot(koli, lidf)  # extinction coefficient in direction of observer
    bf = np.dot(bfli, lidf)
    sob = np.dot(sobli, lidf)
    sof = np.dot(sofli, lidf)

    # 1.3 geometric factors to be used with rho and tau
    sdb = 0.5 * (k + bf)
    sdf = 0.5 * (k - bf)
    ddb = 0.5 * (1 + bf)
    ddf = 0.5 * (1 - bf)
    dob = 0.5 * (K + bf)
    dof = 0.5 * (K - bf)

    # 1.4 solar irradiance factor for all leaf orientations
    Css = cos_ttli * cos_tts  # [nli]
    Ss = sin_ttli * sin_tts  # [nli]
    cos_deltas = np.outer(Css, np.ones(nlazi)) + np.outer(Ss, cos_ttlo)  # [nli, nlazi]
    fs = np.abs(cos_deltas / cos_tts)  # [nli, nlazi]

    # 2. Calculation of reflectance
    # 2.1 reflectance, transmittance factors in a thin layer
    sigb = ddb * rho + ddf * tau  # [nwl]
    sigf = ddf * rho + ddb * tau  # [nwl]
    sb = sdb * rho + sdf * tau  # [nwl]
    sf = sdf * rho + sdb * tau  # [nwl]
    vb = dob * rho + dof * tau  # [nwl]
    vf = dof * rho + dob * tau  # [nwl]
    w = sob * rho + sof * tau  # [nwl]
    a = 1 - sigf  # [nwl]

    # 3. Flux calculation
    # Layer transmittances and reflectances
    tau_ss = np.full(nl, 1 - k * iLAI)
    tau_dd = np.tile(1 - a * iLAI, (nl, 1))
    tau_sd = np.tile(sf * iLAI, (nl, 1))
    rho_sd = np.tile(sb * iLAI, (nl, 1))
    rho_dd = np.tile(sigb * iLAI, (nl, 1))

    # Calculate reflectances (use full spectrum nwl = 2162)
    R_sd, R_dd, Xss, Xsd, Xdd = calc_reflectances(
        tau_ss, tau_sd, tau_dd, rho_dd, rho_sd, rs, nl, nwl
    )

    rdd = R_dd[0, :]
    rsd = R_sd[0, :]

    # Calculate TOC irradiance (use full spectrum wl = wlS, nwl = 2162)
    Esun_, Esky_ = calcTOCirr(atmo, meteo, rdd, rsd, wl, nwl)

    # Calculate flux profiles (use full spectrum nwl = 2162)
    Emins_, Eplus_, Es_s = calc_fluxprofile(Esun_, np.zeros(nwl), rs, Xss, Xsd, Xdd, R_sd, R_dd, nl, nwl)
    Emind_, Eplud_, Es_d = calc_fluxprofile(np.zeros(nwl), Esky_, rs, Xss, Xsd, Xdd, R_sd, R_dd, nl, nwl)
    Emin_ = Emins_ + Emind_
    Eplu_ = Eplus_ + Eplud_

    # 1.5 probabilities Ps, Po, Pso
    Ps = np.exp(k * xl * LAI)  # [nl+1]
    Po = np.exp(K * xl * LAI)  # [nl+1]
    Ps[:nl] = Ps[:nl] * (1 - np.exp(-k * LAI * dx)) / (k * LAI * dx)  # Correct for finite dx
    Po[:nl] = Po[:nl] * (1 - np.exp(-K * LAI * dx)) / (K * LAI * dx)  # Correct for finite dx

    q = canopy.hot
    # Use numba-optimized Pso calculation
    Pso = calc_Pso_array(K, k, LAI, q, dso, xl, dx, nl)

    # Correct Pso for rounding errors
    mask = Pso > Po
    Pso[mask] = np.minimum(Po[mask], Ps[mask])
    mask = Pso > Ps
    Pso[mask] = np.minimum(Po[mask], Ps[mask])

    # 3.3 outgoing fluxes, hemispherical and in viewing direction
    # vegetation contribution - diffuse light
    piLocd_ = (np.sum(vb * Po[:nl, np.newaxis] * Emind_[:nl, :], axis=0) +
               np.sum(vf * Po[:nl, np.newaxis] * Eplud_[:nl, :], axis=0)) * iLAI
    # soil contribution - diffuse light
    piLosd_ = rs * Emind_[nl, :] * Po[nl]

    # vegetation contribution - direct solar light
    piLocu_ = (np.sum(vb * Po[:nl, np.newaxis] * Emins_[:nl, :], axis=0) +
               np.sum(vf * Po[:nl, np.newaxis] * Eplus_[:nl, :], axis=0) +
               np.sum(w * Pso[:nl, np.newaxis] * Esun_[np.newaxis, :], axis=0)) * iLAI
    # soil contribution - direct solar light
    piLosu_ = rs * (Emins_[nl, :] * Po[nl] + Esun_ * Pso[nl])

    piLod_ = piLocd_ + piLosd_  # from Esky
    piLou_ = piLocu_ + piLosu_  # from Esun
    piLoc_ = piLocu_ + piLocd_  # from vegetation
    piLos_ = piLosu_ + piLosd_  # from soil
    piLo_ = piLoc_ + piLos_

    Lo_ = piLo_ / np.pi
    rso = piLou_ / np.maximum(Esun_, 1e-10)
    rdo = piLod_ / np.maximum(Esky_, 1e-10)
    Refl = piLo_ / np.maximum(Esky_ + Esun_, 1e-10)

    # Prevent numerical instability in absorption windows
    mask = Esky_ < 1e-4
    Refl[mask] = rso[mask]
    mask = Esky_ < 2e-4 * np.max(Esky_)
    Refl[mask] = rso[mask]

    # 4. Net fluxes
    # 4.1 incident PAR
    wl_m = wl[Ipar] * 1e-9  # nm to m
    P_ = e2phot(wl_m, Esun_[Ipar] + Esky_[Ipar])
    # Convert photons/m²/s to µmol/m²/s by dividing by Avogadro and * 1e6
    # Note: 0.001 factor accounts for spectral data being in mW/m²/nm
    PAR = 0.001 * sint(P_, wlPAR) / AVOGADRO * 1e6  # umol m-2 s-1
    EPAR_ = Esun_[Ipar] + Esky_[Ipar]
    EPAR = 0.001 * sint(EPAR_, wlPAR)

    # 4.2 Absorbed radiation per layer from direct solar (using numba)
    # Prepare 2D arrays for numba functions
    epsc_2d = np.broadcast_to(epsc, (nl, nwl)).copy() if epsc.ndim == 1 else epsc
    kChlrel_2d = np.broadcast_to(kChlrel, (nl, len(kChlrel))).copy() if kChlrel.ndim == 1 else kChlrel
    kCarrel_2d = np.broadcast_to(kCarrel, (nl, len(kCarrel))).copy() if kCarrel.ndim == 1 else kCarrel

    Asun, Pnsun, Rnsun_PAR, Pnsun_Cab, Rnsun_Cab, Pnsun_Car, Rnsun_Car = calc_layer_absorption_direct(
        nl, len(wlP), len(wlPAR),
        Esun_[IwlP], Esun_[Ipar], Esun_,
        epsc_2d, kChlrel_2d[:, IwlP], kCarrel_2d[:, IwlP],
        wl, wlP, wlPAR, wlP * 1e-9, wlPAR * 1e-9,
    )

    # 4.3 total direct radiation per leaf area
    if lite:
        # Simplified: use mean fs
        fs_mean = np.dot(lidf, np.mean(fs, axis=1))
        Rndir = fs_mean * Asun
        Pndir = fs_mean * Pnsun
        Pndir_Cab = fs_mean * Pnsun_Cab
        Rndir_Cab = fs_mean * Rnsun_Cab
        Pndir_Car = fs_mean * Pnsun_Car
        Rndir_Car = fs_mean * Rnsun_Car
        Rndir_PAR = fs_mean * Rnsun_PAR
    else:
        # Full angular resolution: [nli, nlazi, nl]
        Rndir = np.zeros((nli, nlazi, nl))
        Pndir = np.zeros((nli, nlazi, nl))
        Pndir_Cab = np.zeros((nli, nlazi, nl))
        Rndir_Cab = np.zeros((nli, nlazi, nl))
        Pndir_Car = np.zeros((nli, nlazi, nl))
        Rndir_Car = np.zeros((nli, nlazi, nl))
        Rndir_PAR = np.zeros((nli, nlazi, nl))
        for j in range(nl):
            Rndir[:, :, j] = fs * Asun[j]
            Pndir[:, :, j] = fs * Pnsun[j]
            Pndir_Cab[:, :, j] = fs * Pnsun_Cab[j]
            Rndir_Cab[:, :, j] = fs * Rnsun_Cab[j]
            Pndir_Car[:, :, j] = fs * Pnsun_Car[j]
            Rndir_Car[:, :, j] = fs * Rnsun_Car[j]
            Rndir_PAR[:, :, j] = fs * Rnsun_PAR[j]

    # 4.4 total diffuse radiation per leaf area (shaded leaves) - using numba
    Rndif, Pndif, Pndif_Cab, Rndif_Cab, Pndif_Car, Rndif_Car, Rndif_PAR = calc_layer_absorption_diffuse(
        nl, nwl, len(wlP), len(wlPAR),
        Emin_, Eplu_,
        epsc_2d, kChlrel_2d[:, IwlP], kCarrel_2d[:, IwlP],
        wl, wlP, wlPAR, wlP * 1e-9, wlPAR * 1e-9,
    )

    # Soil layer radiation (use full spectrum wl = wlS)
    Rndirsoil = 0.001 * sint(Esun_ * epss, wl)
    Rndifsoil = 0.001 * sint(Emin_[nl, :] * epss, wl)

    # Net radiation per component
    # Shaded leaves (diffuse only)
    Rnhc = Rndif  # [nl]
    Pnhc = Pndif  # [nl]
    Pnhc_Cab = Pndif_Cab  # [nl]
    Rnhc_Cab = Rndif_Cab  # [nl]
    Pnhc_Car = Pndif_Car  # [nl]
    Rnhc_Car = Rndif_Car  # [nl]
    Rnhc_PAR = Rndif_PAR  # [nl]

    # Sunlit leaves (direct + diffuse)
    if lite:
        Rnuc = Rndir + Rndif  # [nl]
        Pnuc = Pndir + Pndif  # [nl]
        Pnuc_Cab = Pndir_Cab + Pndif_Cab  # [nl]
        Rnuc_Cab = Rndir_Cab + Rndif_Cab  # [nl]
        Pnuc_Car = Pndir_Car + Pndif_Car  # [nl]
        Rnuc_Car = Rndir_Car + Rndif_Car  # [nl]
        Rnuc_PAR = Rndir_PAR + Rndif_PAR  # [nl]
    else:
        # [nli, nlazi, nl]
        Rnuc = np.zeros((nli, nlazi, nl))
        Pnuc = np.zeros((nli, nlazi, nl))
        Pnuc_Cab = np.zeros((nli, nlazi, nl))
        Rnuc_Cab = np.zeros((nli, nlazi, nl))
        Pnuc_Car = np.zeros((nli, nlazi, nl))
        Rnuc_Car = np.zeros((nli, nlazi, nl))
        Rnuc_PAR = np.zeros((nli, nlazi, nl))
        for j in range(nl):
            Rnuc[:, :, j] = Rndir[:, :, j] + Rndif[j]
            Pnuc[:, :, j] = Pndir[:, :, j] + Pndif[j]
            Pnuc_Cab[:, :, j] = Pndir_Cab[:, :, j] + Pndif_Cab[j]
            Rnuc_Cab[:, :, j] = Rndir_Cab[:, :, j] + Rndif_Cab[j]
            Pnuc_Car[:, :, j] = Pndir_Car[:, :, j] + Pndif_Car[j]
            Rnuc_Car[:, :, j] = Rndir_Car[:, :, j] + Rndif_Car[j]
            Rnuc_PAR[:, :, j] = Rndir_PAR[:, :, j] + Rndif_PAR[j]

    # Soil
    Rnus = Rndifsoil + Rndirsoil  # sunlit soil
    Rnhs = Rndifsoil  # shaded soil

    # 5. Output
    Eout_ = Eplu_[0, :]  # Full spectrum (2162 bands)
    # Eouto: optical upward radiation (integrate only optical range 400-2400nm)
    Eouto = 0.001 * sint(Eout_[IwlP], wlP)
    # Thermal output not computed in optical RTM (requires RTMt)
    Eoutt = 0.0
    Lot = 0.0

    # Profiles output
    profiles = {}
    if calc_vert_profiles:
        if lite:
            Pnu1d = Pnuc
            Pnu1d_Cab = Pnuc_Cab
            Pnu1d_Car = Pnuc_Car
        else:
            # meanleaf integration over angles
            from ..supporting.meanleaf import meanleaf
            Pnu1d = meanleaf(Pnuc, lidf, nlazi, 'angles')
            Pnu1d_Cab = meanleaf(Pnuc_Cab, lidf, nlazi, 'angles')
            Pnu1d_Car = meanleaf(Pnuc_Car, lidf, nlazi, 'angles')

        profiles['Pn1d'] = (1 - Ps[:nl]) * Pnhc + Ps[:nl] * Pnu1d
        profiles['Pn1d_Cab'] = (1 - Ps[:nl]) * Pnhc_Cab + Ps[:nl] * Pnu1d_Cab
        profiles['Pn1d_Car'] = (1 - Ps[:nl]) * Pnhc_Car + Ps[:nl] * Pnu1d_Car

    # Create output structures
    gap = GapProbabilities(
        k=k,
        K=K,
        Ps=Ps,
        Po=Po,
        Pso=Pso,
    )

    rad = RadiativeTransferOutput(
        rsd=rsd,
        rdd=rdd,
        rdo=rdo,
        rso=rso,
        refl=Refl,
        rho_dd=rho_dd,
        tau_dd=tau_dd,
        rho_sd=rho_sd,
        tau_ss=tau_ss,
        tau_sd=tau_sd,
        R_sd=R_sd,
        R_dd=R_dd,
        vb=vb,
        vf=vf,
        sigf=sigf,
        sigb=sigb,
        Xdd=Xdd,
        Xsd=Xsd,
        Xss=Xss,
        Esun_=Esun_,
        Esky_=Esky_,
        Emin_=Emin_,
        Eplu_=Eplu_,
        Emins_=Emins_,
        Emind_=Emind_,
        Eplus_=Eplus_,
        Eplud_=Eplud_,
        Lo_=Lo_,
        Eout_=Eout_,
        PAR=PAR,
        EPAR=EPAR,
        Eouto=Eouto,
        Eoutt=Eoutt,
        Lot=Lot,
        Rnhc=Rnhc,
        Pnh=Pnhc * 1e6,  # Convert to umol
        Pnh_Cab=Pnhc_Cab * 1e6,
        Rnh_Cab=Rnhc_Cab,
        Pnh_Car=Pnhc_Car * 1e6,
        Rnh_Car=Rnhc_Car,
        Rnh_PAR=Rnhc_PAR,
        Rnuc=Rnuc,
        Pnu=Pnuc * 1e6 if lite else Pnuc * 1e6,  # Convert to umol
        Pnu_Cab=Pnuc_Cab * 1e6,
        Rnu_Cab=Rnuc_Cab,
        Pnu_Car=Pnuc_Car * 1e6,
        Rnu_Car=Rnuc_Car,
        Rnu_PAR=Rnuc_PAR,
        Rnhs=Rnhs,
        Rnus=Rnus,
    )

    return rad, gap, profiles
