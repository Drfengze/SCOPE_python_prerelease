"""Thermal Radiative Transfer Model (RTMt).

Translated from: src/RTMs/RTMt_sb.m

This module calculates thermal radiation emitted by vegetation and soil,
using the Stefan-Boltzmann (broadband) approach.

References:
    Verhoef, W. (1998). Theory of radiative transfer models applied in
    optical remote sensing of vegetation canopies.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..constants import CONSTANTS
from ..supporting.physics import planck


@dataclass
class ThermalOutput:
    """Output from thermal RTM.

    Attributes:
        Emint: Downwelling thermal irradiance per layer [W m-2]
        Eplut: Upwelling thermal irradiance per layer [W m-2]
        Eoutte: Total upwelling thermal flux at top [W m-2]
        Rnuct: Net thermal radiation for sunlit leaves [W m-2]
        Rnhct: Net thermal radiation for shaded leaves per layer [W m-2]
        Rnust: Net thermal radiation for sunlit soil [W m-2]
        Rnhst: Net thermal radiation for shaded soil [W m-2]
        Lote: Directional thermal radiance (broadband) [W m-2 sr-1]
        Lot_: Spectral directional thermal radiance [W m-2 sr-1 nm-1]
        Eoutte_: Spectral upwelling thermal flux [W m-2 nm-1]
        LST: Land surface temperature [K]
    """
    Emint: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Eplut: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Eoutte: float = 0.0
    Rnuct: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Rnhct: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Rnust: float = 0.0
    Rnhst: float = 0.0
    Lote: float = 0.0
    Lot_: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Eoutte_: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    LST: float = 0.0


def stefan_boltzmann(T, constants=None):
    """Calculate Stefan-Boltzmann thermal emission.

    Args:
        T: Temperature [C] (converted to K internally)
        constants: Physical constants (optional)

    Returns:
        Thermal flux [W m-2]
    """
    sigmaSB = constants.sigmaSB if constants else CONSTANTS.sigmaSB
    T_K = T + 273.15  # Convert to Kelvin
    return sigmaSB * T_K ** 4


def brightness_temperature(radiance, constants=None):
    """Calculate brightness temperature from thermal radiance.

    Args:
        radiance: Thermal radiance [W m-2 sr-1] (includes pi factor)
        constants: Physical constants (optional)

    Returns:
        Brightness temperature [K]
    """
    sigmaSB = constants.sigmaSB if constants else CONSTANTS.sigmaSB
    return (radiance / sigmaSB) ** 0.25


def rtmt_sb(
    rad,
    soil,
    leafbio,
    canopy,
    gap,
    Tcu: NDArray[np.float64],
    Tch: NDArray[np.float64],
    Tsu: float,
    Tsh: float,
    constants=None,
    obsdir: bool = False,
    spectral=None,
    Rli: float = 0.0,
) -> ThermalOutput:
    """Calculate thermal radiation using Stefan-Boltzmann approach.

    Translated from MATLAB RTMt_sb.m

    This is the faster, broadband approach for thermal calculations.

    Args:
        rad: Radiation structure from RTMo with Xdd, Xsd, R_dd, R_sd, etc.
        soil: Soil structure with rs_thermal
        leafbio: Leaf biochemistry structure with rho_thermal, tau_thermal
        canopy: Canopy structure with nlayers, LAI, lidf
        gap: Gap probabilities with Ps, Po, Pso, K
        Tcu: Sunlit leaf temperature [C] - (nl,) or (nli, nlazi, nl)
        Tch: Shaded leaf temperature [C] - (nl,)
        Tsu: Sunlit soil temperature [C]
        Tsh: Shaded soil temperature [C]
        constants: Physical constants
        obsdir: Calculate directional output
        spectral: Spectral bands for directional calculation
        Rli: Incoming longwave radiation from atmosphere [W/m²]

    Returns:
        ThermalOutput with thermal radiation components
    """
    if constants is None:
        constants = CONSTANTS

    output = ThermalOutput()

    # Extract parameters
    nl = canopy.nlayers
    lidf = canopy.lidf
    Ps = gap.Ps
    LAI = canopy.LAI
    dx = 1.0 / nl
    iLAI = LAI * dx

    # Thermal optical properties
    rho = leafbio.rho_thermal if hasattr(leafbio, 'rho_thermal') else 0.01
    tau = leafbio.tau_thermal if hasattr(leafbio, 'tau_thermal') else 0.01
    rs = soil.rs_thermal if hasattr(soil, 'rs_thermal') else 0.05
    epsc = 1 - rho - tau  # Leaf emissivity
    epss = 1 - rs  # Soil emissivity

    # Get reflectance/transmittance arrays from rad (use last wavelength for thermal)
    if hasattr(rad, 'Xdd') and rad.Xdd is not None:
        if rad.Xdd.ndim > 1:
            Xdd = rad.Xdd[:, -1]
            Xsd = rad.Xsd[:, -1]
            R_dd = rad.R_dd[:, -1]
            R_sd = rad.R_sd[:, -1]
            rho_dd = rad.rho_dd[:, -1]
            tau_dd = rad.tau_dd[:, -1]
        else:
            Xdd = rad.Xdd
            Xsd = rad.Xsd
            R_dd = rad.R_dd
            R_sd = rad.R_sd
            rho_dd = rad.rho_dd
            tau_dd = rad.tau_dd
        Xss = np.full(nl, rad.Xss[0] if hasattr(rad.Xss, '__len__') else rad.Xss)
    else:
        # Default values if not available
        Xdd = np.full(nl, 1 - (1 - rho - tau) * iLAI)
        Xsd = np.full(nl, tau * iLAI)
        R_dd = np.full(nl + 1, rs)
        R_sd = np.full(nl + 1, rs)
        rho_dd = np.full(nl, rho * iLAI)
        tau_dd = np.full(nl, 1 - (1 - tau) * iLAI)
        Xss = np.full(nl, 1.0)

    # 1. Calculate radiance by components
    # Stefan-Boltzmann emission
    Hcsu3 = epsc * stefan_boltzmann(Tcu, constants)  # Sunlit leaves
    Hcsh = epsc * stefan_boltzmann(Tch, constants)  # Shaded leaves
    Hssu = epss * stefan_boltzmann(Tsu, constants)  # Sunlit soil
    Hssh = epss * stefan_boltzmann(Tsh, constants)  # Shaded soil

    # Average sunlit leaf emission over angles (if 3D)
    if Hcsu3.ndim == 3:
        # Shape: (nli, nlazi, nl) -> (nl,)
        nli, nlazi = Hcsu3.shape[:2]
        Hcsu = np.zeros(nl)
        for j in range(nl):
            # Weight by LIDF and average over azimuth
            Hcsu[j] = np.sum(Hcsu3[:, :, j] * lidf[:, np.newaxis]) / nlazi
    else:
        Hcsu = Hcsu3

    # Total hemispherical emission by leaf layers
    Hc = Hcsu * Ps[:nl] + Hcsh * (1 - Ps[:nl])

    # Hemispherical emission by soil
    Hs = Hssu * Ps[nl] + Hssh * (1 - Ps[nl])

    # 1.3 Diffuse radiation - upward and downward fluxes
    U = np.zeros(nl + 1)
    Y = np.zeros(nl)
    Es_ = np.zeros(nl + 1)
    Emin = np.zeros(nl + 1)
    Eplu = np.zeros(nl + 1)

    U[nl] = Hs
    Es_[0] = 0
    Emin[0] = 0  # Atmospheric thermal is handled in RTMo, not here

    # Bottom to top
    for j in range(nl - 1, -1, -1):
        if j < len(rho_dd):
            rho_j = rho_dd[j]
            tau_j = tau_dd[j]
            R_dd_j1 = R_dd[j + 1] if j + 1 < len(R_dd) else rs
        else:
            rho_j = rho * iLAI
            tau_j = 1 - (1 - tau) * iLAI
            R_dd_j1 = rs

        denom = 1 - rho_j * R_dd_j1
        if abs(denom) > 1e-10:
            Y[j] = (rho_j * U[j + 1] + Hc[j] * iLAI) / denom
        else:
            Y[j] = Hc[j] * iLAI

        U[j] = tau_j * (R_dd_j1 * Y[j] + U[j + 1]) + Hc[j] * iLAI

    # Top to bottom
    for j in range(nl):
        if j < len(Xss):
            Xss_j = Xss[j]
            Xsd_j = Xsd[j] if j < len(Xsd) else 0
            Xdd_j = Xdd[j] if j < len(Xdd) else 1
            R_sd_j = R_sd[j] if j < len(R_sd) else rs
            R_dd_j = R_dd[j] if j < len(R_dd) else rs
        else:
            Xss_j = 1.0
            Xsd_j = 0
            Xdd_j = 1
            R_sd_j = rs
            R_dd_j = rs

        Es_[j + 1] = Xss_j * Es_[j]
        Emin[j + 1] = Xsd_j * Es_[j] + Xdd_j * Emin[j] + Y[j]
        Eplu[j] = R_sd_j * Es_[j] + R_dd_j * Emin[j] + U[j]

    # MATLAB: Eplu(nl+1) = R_sd(nl)*Es_(nl) + R_dd(nl)*Emin(nl) + Hs
    # R_sd(nl) in MATLAB (1-based) = R_sd[nl-1] in Python (0-based)
    Eplu[nl] = R_sd[nl - 1] * Es_[nl - 1] + R_dd[nl - 1] * Emin[nl - 1] + Hs
    Eoutte = Eplu[0]

    # Store fluxes
    output.Emint = Emin
    output.Eplut = Eplu
    output.Eoutte = Eoutte

    # 2. Total net fluxes
    # Net radiation per component
    if Hcsu3.ndim == 3:
        # Full angular resolution
        nli, nlazi = Hcsu3.shape[:2]
        Rnuct = np.zeros((nli, nlazi, nl))
        for j in range(nl):
            Rnuct[:, :, j] = epsc * (Emin[j] + Eplu[j + 1]) - 2 * Hcsu3[:, :, j]
    else:
        # Lite mode
        Rnuct = np.zeros(nl)
        for j in range(nl):
            Rnuct[j] = epsc * (Emin[j] + Eplu[j + 1]) - 2 * Hcsu[j]

    # Shaded leaves (always 1D)
    Rnhct = np.zeros(nl)
    for j in range(nl):
        Rnhct[j] = epsc * (Emin[j] + Eplu[j + 1]) - 2 * Hcsh[j]

    output.Rnuct = Rnuct
    output.Rnhct = Rnhct

    # Soil net radiation
    # Note: MATLAB uses epss*(Emin - Hssu) where Hssu = epss*sigma*T^4
    # This effectively gives epss^2 for the emission term
    output.Rnust = epss * (Emin[nl] - Hssu)
    output.Rnhst = epss * (Emin[nl] - Hssh)

    # 1.4 Directional radiation and brightness temperature
    if obsdir and hasattr(gap, 'Po') and hasattr(gap, 'K'):
        K = gap.K
        Po = gap.Po
        Pso = gap.Pso

        if hasattr(rad, 'vb') and rad.vb is not None:
            vb = rad.vb[-1] if hasattr(rad.vb, '__len__') else rad.vb
            vf = rad.vf[-1] if hasattr(rad.vf, '__len__') else rad.vf
        else:
            vb = 0.5
            vf = 0.5

        # Directional emitted radiation by vegetation
        piLov = iLAI * (
            K * np.dot(Hcsh, Po[:nl] - Pso[:nl]) +
            K * np.dot(Hcsu, Pso[:nl]) +
            np.dot(vb * Emin[:nl] + vf * Eplu[:nl], Po[:nl])
        )

        # Directional emitted radiation by soil
        piLos = Hssh * (Po[nl] - Pso[nl]) + Hssu * Pso[nl]

        piLot = piLov + piLos
        output.Lote = piLot / np.pi

        # Brightness temperature
        Tbr = (piLot / constants.sigmaSB) ** 0.25
        output.LST = Tbr

        if spectral is not None and hasattr(spectral, 'wlS'):
            output.Lot_ = planck(spectral.wlS, Tbr)
            Tbr2 = (Eoutte / constants.sigmaSB) ** 0.25
            output.Eoutte_ = planck(spectral.wlS, Tbr2)

    return output


def rtmt_planck(
    spectral,
    rad,
    soil,
    leafopt,
    canopy,
    gap,
    Tcu: NDArray[np.float64],
    Tch: NDArray[np.float64],
    Tsu: float,
    Tsh: float,
):
    """Calculate thermal radiation using Planck's law for each wavelength.

    Translated from MATLAB RTMt_planck.m

    This is the spectrally-resolved approach for thermal calculations,
    computing Planck radiation at each thermal wavelength.

    Args:
        spectral: Spectral bands with IwlT, wlT indices
        rad: Radiation structure from RTMo with Xdd, Xsd, R_dd, R_sd, etc.
        soil: Soil structure with refl spectrum
        leafopt: Leaf optical properties with refl, tran spectra
        canopy: Canopy structure with nlayers, LAI, lidf
        gap: Gap probabilities with Ps, Po, Pso, K
        Tcu: Sunlit leaf temperature [C] - (nl,) or (nli, nlazi, nl)
        Tch: Shaded leaf temperature [C] - (nl,)
        Tsu: Sunlit soil temperature [C]
        Tsh: Shaded soil temperature [C]

    Returns:
        Updated rad structure with thermal components
    """
    # Extract thermal wavelength indices
    IT = spectral.IwlT
    wlt = spectral.wlT
    nwlt = len(IT)

    # Canopy parameters
    nl = canopy.nlayers
    lidf = canopy.lidf
    Ps = gap.Ps
    LAI = canopy.LAI
    dx = 1.0 / nl
    iLAI = LAI * dx

    # Optical properties at thermal wavelengths
    # leafopt.refl and leafopt.tran are [nwl] or [nl, nwl]
    if hasattr(leafopt, 'refl') and leafopt.refl is not None:
        if leafopt.refl.ndim > 1:
            rho = leafopt.refl[:, IT].T  # [nwlt, nl] or [nwlt]
            tau = leafopt.tran[:, IT].T
        else:
            rho = leafopt.refl[IT]  # [nwlt]
            tau = leafopt.tran[IT]
    else:
        rho = np.full(nwlt, 0.01)
        tau = np.full(nwlt, 0.01)

    # Soil reflectance at thermal wavelengths
    if hasattr(soil, 'refl') and soil.refl is not None:
        rs = soil.refl[IT]
    else:
        rs = np.full(nwlt, 0.05)

    epsc = 1 - rho - tau  # Leaf emissivity [nwlt]
    epss = 1 - rs  # Soil emissivity [nwlt]

    # Radiative transfer coefficients at thermal wavelengths
    Xdd = rad.Xdd[:, IT]  # [nl, nwlt]
    Xsd = rad.Xsd[:, IT]
    # Xss is per layer (shape (nl,)), not per wavelength - tile to [nl, nwlt]
    if hasattr(rad.Xss, '__len__') and rad.Xss.ndim == 1:
        # rad.Xss is [nl], tile across wavelengths
        Xss = np.tile(rad.Xss[:, np.newaxis], (1, nwlt))  # [nl, nwlt]
    elif hasattr(rad.Xss, '__len__') and rad.Xss.ndim == 2:
        # rad.Xss is [nl, nwl], extract thermal
        Xss = rad.Xss[:, IT]  # [nl, nwlt]
    else:
        Xss = np.full((nl, nwlt), rad.Xss)
    R_dd = rad.R_dd[:, IT]  # [nl+1, nwlt]
    R_sd = rad.R_sd[:, IT]
    rho_dd = rad.rho_dd[:, IT]
    tau_dd = rad.tau_dd[:, IT]

    # Initialize output arrays
    piLot_ = np.zeros(nwlt)
    Eoutte_ = np.zeros(nwlt)
    Emin_ = np.zeros((nl + 1, nwlt))
    Eplu_ = np.zeros((nl + 1, nwlt))

    # Loop over thermal wavelengths
    for i in range(nwlt):
        # Emissivity at this wavelength
        epsc_i = epsc[i] if epsc.ndim == 1 else epsc[i, :]
        epss_i = epss[i]

        # 1.1 Radiance by components using Planck function
        # pi * Planck(wl, T, eps) gives spectral radiant exitance
        Hcsu3 = np.pi * planck(wlt[i], Tcu + 273.15) * epsc_i
        Hcsh = np.pi * planck(wlt[i], Tch + 273.15) * epsc_i
        Hssu = np.pi * planck(wlt[i], Tsu + 273.15) * epss_i
        Hssh = np.pi * planck(wlt[i], Tsh + 273.15) * epss_i

        # 1.2 Average sunlit leaf emission over angles (if 3D)
        if Hcsu3.ndim == 3:
            nli, nlazi = Hcsu3.shape[:2]
            v1 = np.ones(nlazi) / nlazi
            Hcsu = np.zeros(nl)
            for j in range(nl):
                Hcsu2 = Hcsu3[:, :, j]  # [nli, nlazi]
                Hcsu[j] = np.dot(v1, np.dot(Hcsu2.T, lidf))
        else:
            Hcsu = Hcsu3

        # Hemispherical emission by layers
        Hc = Hcsu * Ps[:nl] + Hcsh * (1 - Ps[:nl])
        Hs = Hssu * Ps[nl] + Hssh * (1 - Ps[nl])

        # 1.3 Diffuse radiation
        U = np.zeros(nl + 1)
        Y = np.zeros(nl)
        Es_ = np.zeros(nl + 1)
        Emin = np.zeros(nl + 1)
        Eplu = np.zeros(nl + 1)

        U[nl] = Hs
        Es_[0] = 0
        Emin[0] = 0

        # Bottom to top
        for j in range(nl - 1, -1, -1):
            denom = 1 - rho_dd[j, i] * R_dd[j + 1, i]
            if abs(denom) > 1e-10:
                Y[j] = (rho_dd[j, i] * U[j + 1] + Hc[j] * iLAI) / denom
            else:
                Y[j] = Hc[j] * iLAI
            U[j] = tau_dd[j, i] * (R_dd[j + 1, i] * Y[j] + U[j + 1]) + Hc[j] * iLAI

        # Top to bottom
        for j in range(nl):
            Es_[j + 1] = Xss[j, i] * Es_[j] if Xss.ndim > 1 else Xss * Es_[j]
            Emin[j + 1] = Xsd[j, i] * Es_[j] + Xdd[j, i] * Emin[j] + Y[j]
            Eplu[j] = R_sd[j, i] * Es_[j] + R_dd[j, i] * Emin[j] + U[j]

        Eplu[nl] = R_sd[nl - 1, i] * Es_[nl - 1] + R_dd[nl - 1, i] * Emin[nl - 1] + Hs

        # Store fluxes
        Emin_[:, i] = Emin
        Eplu_[:, i] = Eplu
        Eoutte_[i] = Eplu[0]

        # 1.4 Directional radiation
        K = gap.K
        vb = rad.vb[-1] if hasattr(rad.vb, '__len__') else rad.vb
        vf = rad.vf[-1] if hasattr(rad.vf, '__len__') else rad.vf

        # Directional emitted radiation by vegetation
        piLov = iLAI * (
            K * np.dot(Hcsh, gap.Po[:nl] - gap.Pso[:nl]) +
            K * np.dot(Hcsu, gap.Pso[:nl]) +
            np.dot(vb * Emin[:nl] + vf * Eplu[:nl], gap.Po[:nl])
        )

        # Directional emitted radiation by soil
        piLos = Hssh * (gap.Po[nl] - gap.Pso[nl]) + Hssu * gap.Pso[nl]

        piLot_[i] = piLov + piLos

    Lot_ = piLot_ / np.pi

    # Write output to rad structure
    nwlS = len(spectral.wlS)
    rad.Lot_ = np.zeros(nwlS)
    rad.Eoutte_ = np.zeros(nwlS)
    rad.Lot_[IT] = Lot_
    rad.Eoutte_[IT] = Eoutte_
    rad.Eplut_ = Eplu_
    rad.Emint_ = Emin_

    return rad
