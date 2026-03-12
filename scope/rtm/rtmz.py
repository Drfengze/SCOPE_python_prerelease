"""Xanthophyll/PRI Radiative Transfer Model (RTMz).

Translated from: src/RTMs/RTMz.m

RTMz calculates the modification of TOC outgoing radiance due to the
xanthophyll cycle (violaxanthin to zeaxanthin conversion). The
Photochemical Reflectance Index (PRI) is sensitive to this conversion.

References:
    Vilfan, N., et al. (2018). Extending Fluspect to simulate xanthophyll
    driven leaf reflectance dynamics.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..constants import CONSTANTS
from ..supporting.integration import sint


@dataclass
class XanthophyllOutput:
    """Output from xanthophyll/PRI RTM.

    Attributes:
        Lo_mod: Modified spectral radiance [W m-2 sr-1 nm-1]
        refl_mod: Modified reflectance
        rso_mod: Modified directional reflectance (sun-observer)
        rdo_mod: Modified hemispherical-directional reflectance
        Eout_mod: Modified hemispherical outgoing radiation
        delta_Lo: Change in radiance due to xanthophyll conversion
        delta_refl: Change in reflectance
        PRI: Photochemical Reflectance Index
    """
    Lo_mod: np.ndarray = field(default_factory=lambda: np.array([]))
    refl_mod: np.ndarray = field(default_factory=lambda: np.array([]))
    rso_mod: np.ndarray = field(default_factory=lambda: np.array([]))
    rdo_mod: np.ndarray = field(default_factory=lambda: np.array([]))
    Eout_mod: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_Lo: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_refl: np.ndarray = field(default_factory=lambda: np.array([]))
    PRI: float = 0.0


def Kn2Cx(Kn: np.ndarray) -> np.ndarray:
    """Convert NPQ rate constant to xanthophyll conversion factor.

    Empirical relationship from Vilfan et al. (2018, 2019).

    Args:
        Kn: Non-photochemical quenching rate constant

    Returns:
        Cx: Xanthophyll conversion factor (0-1)
    """
    # Empirical fit: Cx = 0.3187 * Kn (Vilfan et al., 2018, 2019)
    Cx = 0.3187 * np.asarray(Kn)
    return np.clip(Cx, 0.0, 1.0)


def rtmz(
    spectral,
    rad,
    soil,
    leafopt,
    canopy,
    gap,
    angles,
    Knu: NDArray[np.float64],
    Knh: NDArray[np.float64],
) -> XanthophyllOutput:
    """Calculate xanthophyll-induced modification to canopy reflectance.

    This implementation follows the MATLAB RTMz.m code structure.

    Args:
        spectral: Spectral band definitions with wlS, wlZ
        rad: Radiative transfer output from RTMo
        soil: Soil properties with reflectance
        leafopt: Leaf optical with reflZ, tranZ (zeaxanthin spectra)
        canopy: Canopy structure (LAI, nlayers, lidf, litab, lazitab)
        gap: Gap probabilities (Ps, Po, Pso)
        angles: Viewing geometry (tts, tto, psi)
        Knu: NPQ rate constant for sunlit leaves (nl,) or (nli, nlazi, nl)
        Knh: NPQ rate constant for shaded leaves (nl,)

    Returns:
        XanthophyllOutput with modified reflectance/radiance
    """
    output = XanthophyllOutput()

    # Check if xanthophyll spectra exist
    if not hasattr(leafopt, 'reflZ') or leafopt.reflZ is None:
        # Return unmodified values
        output.Lo_mod = rad.Lo_.copy() if rad.Lo_ is not None else np.array([])
        output.refl_mod = rad.refl.copy() if rad.refl is not None else np.array([])
        return output

    # Wavelength setup
    wlS = spectral.wlS
    wlZ = spectral.wlZ  # Xanthophyll wavelengths (500-600 nm)

    # Find indices of Z wavelengths in wlS
    iwlfi = np.array([np.argmin(np.abs(wlS - w)) for w in wlZ])
    nwlZ = len(iwlfi)

    nl = canopy.nlayers
    LAI = canopy.LAI
    iLAI = LAI / nl

    # Leaf angle tables
    litab = canopy.litab
    lazitab = canopy.lazitab
    lidf = canopy.lidf
    nlazi = len(lazitab)
    nlinc = len(litab)
    nlori = nlinc * nlazi

    # Gap probabilities
    Ps = gap.Ps
    Po = gap.Po
    Pso = gap.Pso
    Qs = Ps[:nl]

    # Reflectance/transmittance change due to V->Z conversion
    # RZ = reflZ - refl (change when fully converted to zeaxanthin)
    RZ = (leafopt.reflZ[iwlfi] - leafopt.refl[iwlfi]).T if leafopt.reflZ.ndim == 1 else (leafopt.reflZ[:, iwlfi] - leafopt.refl[:, iwlfi]).T
    TZ = (leafopt.tranZ[iwlfi] - leafopt.tran[iwlfi]).T if leafopt.tranZ.ndim == 1 else (leafopt.tranZ[:, iwlfi] - leafopt.tran[:, iwlfi]).T

    # Extract spectral fluxes at Z wavelengths
    Esunf_ = rad.Esun_[iwlfi]

    # Handle direct and diffuse separately (matching MATLAB)
    if hasattr(rad, 'Emins_') and rad.Emins_ is not None:
        Eminf_s = rad.Emins_[:, iwlfi].T  # Direct contribution
        Eminf_d = rad.Emind_[:, iwlfi].T  # Diffuse contribution
        Epluf_s = rad.Eplus_[:, iwlfi].T
        Epluf_d = rad.Eplud_[:, iwlfi].T
    else:
        # Fallback: use single Emin_/Eplu_
        Eminf_s = rad.Emin_[:, iwlfi].T if rad.Emin_.ndim == 2 else np.tile(rad.Emin_[iwlfi][:, np.newaxis], (1, nl + 1))
        Eminf_d = Eminf_s * 0
        Epluf_s = rad.Eplu_[:, iwlfi].T if rad.Eplu_.ndim == 2 else np.tile(rad.Eplu_[iwlfi][:, np.newaxis], (1, nl + 1))
        Epluf_d = Epluf_s * 0

    # RTM coefficients at Z wavelengths
    def safe_get(arr, idx, nl):
        if arr is None:
            return np.zeros((nl, len(idx)))
        if arr.ndim == 2:
            return arr[:, idx]
        return np.tile(arr[idx], (nl, 1))

    Xdd = safe_get(rad.Xdd, iwlfi, nl)
    rho_dd = safe_get(rad.rho_dd, iwlfi, nl)
    R_dd = safe_get(rad.R_dd, iwlfi, nl + 1) if hasattr(rad, 'R_dd') else np.zeros((nl + 1, nwlZ))
    tau_dd = safe_get(rad.tau_dd, iwlfi, nl)
    vb = safe_get(rad.vb, iwlfi, nl)
    vf = safe_get(rad.vf, iwlfi, nl)

    # Geometric quantities
    deg2rad = CONSTANTS.deg2rad
    tto = angles.tto
    tts = angles.tts
    psi = angles.psi

    rs = soil.refl[iwlfi]

    cos_tto = np.cos(tto * deg2rad)
    sin_tto = np.sin(tto * deg2rad)
    cos_tts = np.cos(tts * deg2rad)
    sin_tts = np.sin(tts * deg2rad)

    cos_ttli = np.cos(litab * deg2rad)
    sin_ttli = np.sin(litab * deg2rad)
    cos_phils = np.cos(lazitab * deg2rad)
    cos_philo = np.cos((lazitab - psi) * deg2rad)

    # Geometric factors
    cds = np.outer(cos_ttli, np.ones(nlazi)) * cos_tts + np.outer(sin_ttli, cos_phils) * sin_tts
    cdo = np.outer(cos_ttli, np.ones(nlazi)) * cos_tto + np.outer(sin_ttli, cos_philo) * sin_tto

    fs = cds / cos_tts
    fo = cdo / cos_tto

    absfs = np.abs(fs).flatten()
    absfo = np.abs(fo).flatten()
    fsfo = (fs * fo).flatten()
    absfsfo = np.abs(fsfo)
    foctl = (fo * np.outer(cos_ttli, np.ones(nlazi))).flatten()
    fsctl = (fs * np.outer(cos_ttli, np.ones(nlazi))).flatten()
    ctl2 = np.outer(cos_ttli ** 2, np.ones(nlazi)).flatten()

    # Initialize output flux arrays
    Fmin_ = np.zeros((nl + 1, nwlZ, 2))
    Fplu_ = np.zeros((nl + 1, nwlZ, 2))
    LoF_ = np.zeros((nwlZ, 2))

    laz = 1.0 / 36

    # Convert Kn to Cx
    etah = Kn2Cx(Knh)

    # Handle Knu dimensions
    if Knu.ndim == 1:
        etau = np.tile(Kn2Cx(Knu)[np.newaxis, :], (nlori, 1))
    elif Knu.ndim == 3:
        etau = Kn2Cx(Knu).reshape(nlori, nl)
    else:
        etau = np.tile(Kn2Cx(Knu).flatten()[:, np.newaxis], (1, nl))

    # Ensure correct shape
    if etau.shape[0] != nlori:
        etau = np.tile(etau[:1, :], (nlori, 1))
    if etau.shape[1] != nl:
        etau = np.tile(etau[:, :1], (1, nl))

    # Weight by LIDF
    lidf_laz = np.tile(lidf * laz, nlazi)
    etau_lidf = etau * lidf_laz[:, np.newaxis]

    if etah.ndim == 1:
        etah_flat = np.tile(etah[np.newaxis, :], (nlori, 1))
    else:
        etah_flat = np.tile(etah.flatten()[:, np.newaxis], (1, nl))

    if etah_flat.shape[0] != nlori:
        etah_flat = np.tile(etah_flat[:1, :], (nlori, 1))
    if etah_flat.shape[1] != nl:
        etah_flat = np.tile(etah_flat[:, :1], (1, nl))

    etah_lidf = etah_flat * lidf_laz[:, np.newaxis]

    # Loop over direct (k=0) and diffuse (k=1) components
    for k in range(2):
        U = np.zeros((nl + 1, nwlZ))
        Y = np.zeros((nl, nwlZ))

        # Create effective reflectance/transmittance modification
        if k == 0:  # Direct (includes sun)
            MpluEsun = RZ * Esunf_
            MminEsun = TZ * Esunf_
            Eminf_ = Eminf_s
            Epluf_ = Epluf_s
        else:  # Diffuse only
            MpluEsun = np.zeros_like(RZ)
            MminEsun = np.zeros_like(TZ)
            Eminf_ = Eminf_d
            Epluf_ = Epluf_d

        # Weighted emission factors
        sum_etau_absfsfo = np.sum(etau_lidf * absfsfo[:, np.newaxis], axis=0)
        sum_etau_fsfo = np.sum(etau_lidf * fsfo[:, np.newaxis], axis=0)
        sum_etau_absfs = np.sum(etau_lidf * absfs[:, np.newaxis], axis=0)
        sum_etau_fsctl = np.sum(etau_lidf * fsctl[:, np.newaxis], axis=0)
        sum_etau_absfo = np.sum(etau_lidf * absfo[:, np.newaxis], axis=0)
        sum_etau_foctl = np.sum(etau_lidf * foctl[:, np.newaxis], axis=0)
        sum_etau = np.sum(etau_lidf, axis=0)
        sum_etau_ctl2 = np.sum(etau_lidf * ctl2[:, np.newaxis], axis=0)

        sum_etah_absfo = np.sum(etah_lidf * absfo[:, np.newaxis], axis=0)
        sum_etah_foctl = np.sum(etah_lidf * foctl[:, np.newaxis], axis=0)
        sum_etah = np.sum(etah_lidf, axis=0)
        sum_etah_ctl2 = np.sum(etah_lidf * ctl2[:, np.newaxis], axis=0)

        # Calculate layer-by-layer modification
        MpluEmin = RZ[:, np.newaxis] * Eminf_[:, :nl]
        MpluEplu = RZ[:, np.newaxis] * Epluf_[:, :nl]
        MminEmin = TZ[:, np.newaxis] * Eminf_[:, :nl]
        MminEplu = TZ[:, np.newaxis] * Epluf_[:, :nl]

        # Emitted flux components
        wfEs = MpluEsun[:, np.newaxis] * sum_etau_absfsfo + MminEsun[:, np.newaxis] * sum_etau_fsfo
        sfEs = MpluEsun[:, np.newaxis] * sum_etau_absfs - MminEsun[:, np.newaxis] * sum_etau_fsctl
        sbEs = MpluEsun[:, np.newaxis] * sum_etau_absfs + MminEsun[:, np.newaxis] * sum_etau_fsctl

        vfEplu_h = MpluEplu * sum_etah_absfo - MminEplu * sum_etah_foctl
        vfEplu_u = MpluEplu * sum_etau_absfo - MminEplu * sum_etau_foctl
        vbEmin_h = MpluEmin * sum_etah_absfo + MminEmin * sum_etah_foctl
        vbEmin_u = MpluEmin * sum_etau_absfo + MminEmin * sum_etau_foctl

        sigfEmin_h = MpluEmin * sum_etah - MminEmin * sum_etah_ctl2
        sigfEmin_u = MpluEmin * sum_etau - MminEmin * sum_etau_ctl2
        sigbEmin_h = MpluEmin * sum_etah + MminEmin * sum_etah_ctl2
        sigbEmin_u = MpluEmin * sum_etau + MminEmin * sum_etau_ctl2

        sigfEplu_h = MpluEplu * sum_etah - MminEplu * sum_etah_ctl2
        sigfEplu_u = MpluEplu * sum_etau - MminEplu * sum_etau_ctl2
        sigbEplu_h = MpluEplu * sum_etah + MminEplu * sum_etah_ctl2
        sigbEplu_u = MpluEplu * sum_etau + MminEplu * sum_etau_ctl2

        # Total emitted flux modification
        piLs = wfEs + vfEplu_u + vbEmin_u
        piLd = vbEmin_h + vfEplu_h
        Fsmin = sfEs + sigfEmin_u + sigbEplu_u
        Fsplu = sbEs + sigbEmin_u + sigfEplu_u
        Fdmin = sigfEmin_h + sigbEplu_h
        Fdplu = sigbEmin_h + sigfEplu_h

        Femmin = iLAI * (Fsmin * Qs + Fdmin * (1 - Qs))
        Femplu = iLAI * (Fsplu * Qs + Fdplu * (1 - Qs))

        # Adding method (bottom to top)
        for j in range(nl - 1, -1, -1):
            denom = 1 - rho_dd[j, :] * R_dd[j + 1, :]
            denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            Y[j, :] = (rho_dd[j, :] * U[j + 1, :] + Femmin[:, j]) / denom
            U[j, :] = tau_dd[j, :] * (R_dd[j + 1, :] * Y[j, :] + U[j + 1, :]) + Femplu[:, j]

        # Propagate through canopy (top to bottom)
        for j in range(nl):
            Fmin_[j + 1, :, k] = Xdd[j, :] * Fmin_[j, :, k] + Y[j, :]
            Fplu_[j, :, k] = R_dd[j, :] * Fmin_[j, :, k] + U[j, :]

        # Calculate observed radiance
        piLo1 = iLAI * (piLs @ Pso[:nl])
        piLo2 = iLAI * (piLd @ (Po[:nl] - Pso[:nl]))
        # Note: transpose AFTER element-wise multiplication to get correct shapes
        piLo3 = iLAI * ((vb * Fmin_[:nl, :, k] + vf * Fplu_[:nl, :, k]).T @ Po[:nl])
        piLo4 = rs * (Po[nl] * Fmin_[nl, :, k])

        piLtot = piLo1 + piLo2 + piLo3 + piLo4
        LoF_[:, k] = piLtot / np.pi

    # Total hemispherical flux modification
    Fhem_ = np.sum(Fplu_[0, :, :], axis=1)

    # Apply modifications to rad outputs
    delta_Lo = np.zeros(len(wlS))
    delta_Lo[iwlfi] = np.sum(LoF_, axis=1)

    delta_Eout = np.zeros(len(wlS))
    delta_Eout[iwlfi] = Fhem_

    # Modified outputs
    output.Lo_mod = rad.Lo_.copy() if rad.Lo_ is not None else np.zeros(len(wlS))
    output.Lo_mod[iwlfi] = output.Lo_mod[iwlfi] + np.sum(LoF_, axis=1)

    output.delta_Lo = delta_Lo

    # Modified rso and rdo
    output.rso_mod = rad.rso.copy() if rad.rso is not None else np.zeros(len(wlS))
    eps = rad.Esun_ + 1e-10
    output.rso_mod[iwlfi] = output.rso_mod[iwlfi] + LoF_[:, 0] / eps[iwlfi]

    output.rdo_mod = rad.rdo.copy() if rad.rdo is not None else np.zeros(len(wlS))
    eps = rad.Esky_ + 1e-10
    output.rdo_mod[iwlfi] = output.rdo_mod[iwlfi] + LoF_[:, 1] / eps[iwlfi]

    # Modified reflectance
    output.refl_mod = rad.refl.copy() if rad.refl is not None else np.zeros(len(wlS))
    eps = rad.Esky_ + rad.Esun_ + 1e-10
    output.refl_mod[iwlfi] = np.pi * output.Lo_mod[iwlfi] / eps[iwlfi]

    output.delta_refl = np.zeros(len(wlS))
    output.delta_refl[iwlfi] = np.pi * delta_Lo[iwlfi] / eps[iwlfi]

    # Modified Eout
    output.Eout_mod = rad.Eout_.copy() if rad.Eout_ is not None else np.zeros(len(wlS))
    output.Eout_mod[iwlfi] = output.Eout_mod[iwlfi] + Fhem_

    # Calculate PRI
    output.PRI = calculate_pri(output.refl_mod, wlS)

    return output


def calculate_pri(
    refl: np.ndarray,
    wl: np.ndarray,
) -> float:
    """Calculate Photochemical Reflectance Index from reflectance spectrum.

    PRI = (R531 - R570) / (R531 + R570)

    Args:
        refl: Reflectance spectrum
        wl: Wavelengths [nm]

    Returns:
        PRI value
    """
    idx_531 = np.argmin(np.abs(wl - 531))
    idx_570 = np.argmin(np.abs(wl - 570))

    R531 = refl[idx_531]
    R570 = refl[idx_570]

    if (R531 + R570) > 1e-10:
        return (R531 - R570) / (R531 + R570)
    else:
        return 0.0
