"""Fluorescence Radiative Transfer Model (RTMf).

Translated from: src/RTMs/RTMf.m

RTMf calculates the spectrum of solar-induced fluorescence (SIF) radiance
in the observer's direction and the TOC spectral hemispherical upward flux.

References:
    Verhoef, W., & van der Tol, C. (2007). Fluorescence radiative transfer.
    Van der Tol, C., et al. (2014). Models of fluorescence and photosynthesis.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from ..constants import CONSTANTS
from ..supporting.integration import sint


@dataclass
class FluorescenceOutput:
    """Output from fluorescence RTM.

    Attributes:
        LoF_: Directional fluorescence radiance [W m-2 sr-1 nm-1]
        EoutF_: Hemispherical fluorescence flux [W m-2 nm-1]
        LoF_sunlit: Sunlit leaf contribution to LoF_
        LoF_shaded: Shaded leaf contribution to LoF_
        LoF_scattered: Scattered fluorescence contribution
        LoF_soil: Soil-reflected fluorescence
        Femleaves_: Total fluorescence emitted by leaves [W m-2 nm-1]
        EoutF: Total hemispherical fluorescence [W m-2]
        LoutF: Total directional fluorescence [W m-2 sr-1]
        F685: Peak fluorescence near 685 nm [W m-2 sr-1 nm-1]
        wl685: Wavelength of 685 nm peak [nm]
        F740: Peak fluorescence near 740 nm [W m-2 sr-1 nm-1]
        wl740: Wavelength of 740 nm peak [nm]
        F684: Fluorescence at 684 nm [W m-2 sr-1 nm-1]
        F761: Fluorescence at 761 nm [W m-2 sr-1 nm-1]
    """
    LoF_: np.ndarray = field(default_factory=lambda: np.array([]))
    EoutF_: np.ndarray = field(default_factory=lambda: np.array([]))
    LoF_sunlit: np.ndarray = field(default_factory=lambda: np.array([]))
    LoF_shaded: np.ndarray = field(default_factory=lambda: np.array([]))
    LoF_scattered: np.ndarray = field(default_factory=lambda: np.array([]))
    LoF_soil: np.ndarray = field(default_factory=lambda: np.array([]))
    Femleaves_: np.ndarray = field(default_factory=lambda: np.array([]))
    EoutF: float = 0.0
    LoutF: float = 0.0
    F685: float = 0.0
    wl685: float = 685.0
    F740: float = 0.0
    wl740: float = 740.0
    F684: float = 0.0
    F761: float = 0.0


def ephoton(wavelength: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate energy content of a single photon.

    Args:
        wavelength: Wavelength in meters

    Returns:
        Energy per photon in Joules
    """
    h = CONSTANTS.h
    c = CONSTANTS.c
    return h * c / wavelength


def e2phot(wavelength: NDArray[np.float64], E: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert energy to moles of photons.

    Args:
        wavelength: Wavelength in meters
        E: Energy in Joules (per m2 per nm)

    Returns:
        Moles of photons
    """
    e = ephoton(wavelength)
    photons = E / e
    return photons / CONSTANTS.A


def rtmf(
    spectral,
    rad,
    soil,
    leafopt,
    canopy,
    gap,
    angles,
    etau: NDArray[np.float64],
    etah: NDArray[np.float64],
) -> FluorescenceOutput:
    """Calculate fluorescence radiative transfer through the canopy.

    This implementation follows the MATLAB RTMf.m code structure.

    Args:
        spectral: Spectral band definitions with wlS, wlF, wlE
        rad: Radiative transfer output from RTMo (Esun_, Emin_, Eplu_, etc.)
        soil: Soil properties with reflectance spectrum
        leafopt: Leaf optical properties with Mb, Mf fluorescence matrices
        canopy: Canopy structure (LAI, nlayers, lidf, litab, lazitab)
        gap: Gap probabilities (Ps, Po, Pso)
        angles: Viewing geometry (tts, tto, psi)
        etau: Fluorescence efficiency for sunlit leaves (nl,) or (nli, nlazi, nl)
        etah: Fluorescence efficiency for shaded leaves (nl,)

    Returns:
        FluorescenceOutput containing fluorescence radiances and fluxes
    """
    output = FluorescenceOutput()

    # Check if fluorescence matrices exist
    if leafopt.Mb is None or leafopt.Mf is None:
        nwlF = len(spectral.wlF)
        output.LoF_ = np.zeros(nwlF)
        output.EoutF_ = np.zeros(nwlF)
        return output

    # Internal wavelength grids (reduced resolution for speed, matching MATLAB)
    wlS = spectral.wlS
    wlF_calc = np.arange(640, 851, 4, dtype=float)  # Internal fluorescence wavelengths
    wlE_calc = np.arange(400, 751, 5, dtype=float)  # Internal excitation wavelengths

    # Find indices of excitation and emission wavelengths in wlS
    iwlfi = np.array([np.argmin(np.abs(wlS - w)) for w in wlE_calc])
    iwlfo = np.array([np.argmin(np.abs(wlS - w)) for w in wlF_calc])

    nf = len(iwlfo)  # Number of fluorescence wavelengths
    nl = canopy.nlayers
    LAI = canopy.LAI

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
    Qs = Ps[:nl]  # Sunlit fraction per layer

    # Initialize flux arrays
    MpluEmin = np.zeros((nf, nl))
    MpluEplu = np.zeros((nf, nl))
    MminEmin = np.zeros((nf, nl))
    MminEplu = np.zeros((nf, nl))
    MpluEsun = np.zeros((nf, nl))
    MminEsun = np.zeros((nf, nl))

    # Extract spectral fluxes at excitation wavelengths
    Esunf_ = rad.Esun_[iwlfi]

    # Handle Emin_ and Eplu_ dimensions
    if rad.Emin_.ndim == 2:
        Eminf_ = rad.Emin_[:, iwlfi].T  # (nwlE, nl+1)
        Epluf_ = rad.Eplu_[:, iwlfi].T
    else:
        Eminf_ = np.tile(rad.Emin_[iwlfi][:, np.newaxis], (1, nl + 1))
        Epluf_ = np.tile(rad.Eplu_[iwlfi][:, np.newaxis], (1, nl + 1))

    iLAI = LAI / nl

    # Extract RTM coefficients at fluorescence wavelengths
    def safe_extract(arr, idx, default_shape):
        if arr is None:
            return np.ones(default_shape) if 'tau' in str(default_shape) else np.zeros(default_shape)
        if arr.ndim == 2:
            return arr[:, idx]
        return np.tile(arr[idx][:, np.newaxis], (1, default_shape[0])).T

    Xdd = safe_extract(rad.Xdd, iwlfo, (nl, nf)) if hasattr(rad, 'Xdd') and rad.Xdd is not None else np.ones((nl, nf))
    rho_dd = safe_extract(rad.rho_dd, iwlfo, (nl, nf)) if hasattr(rad, 'rho_dd') and rad.rho_dd is not None else np.zeros((nl, nf))
    R_dd = safe_extract(rad.R_dd, iwlfo, (nl + 1, nf)) if hasattr(rad, 'R_dd') and rad.R_dd is not None else np.zeros((nl + 1, nf))
    tau_dd = safe_extract(rad.tau_dd, iwlfo, (nl, nf)) if hasattr(rad, 'tau_dd') and rad.tau_dd is not None else np.ones((nl, nf))
    vb = safe_extract(rad.vb, iwlfo, (nl, nf)) if hasattr(rad, 'vb') and rad.vb is not None else np.zeros((nl, nf))
    vf = safe_extract(rad.vf, iwlfo, (nl, nf)) if hasattr(rad, 'vf') and rad.vf is not None else np.zeros((nl, nf))

    # Geometric quantities
    deg2rad = CONSTANTS.deg2rad
    tto = angles.tto
    tts = angles.tts
    psi = angles.psi

    # Soil reflectance at fluorescence wavelengths
    rs = soil.refl[iwlfo]

    cos_tto = np.cos(tto * deg2rad)
    sin_tto = np.sin(tto * deg2rad)
    cos_tts = np.cos(tts * deg2rad)
    sin_tts = np.sin(tts * deg2rad)

    cos_ttli = np.cos(litab * deg2rad)
    sin_ttli = np.sin(litab * deg2rad)
    cos_phils = np.cos(lazitab * deg2rad)
    cos_philo = np.cos((lazitab - psi) * deg2rad)

    # Geometric factors for all leaf angle/azimuth classes
    cds = np.outer(cos_ttli, np.ones(nlazi)) * cos_tts + np.outer(sin_ttli, cos_phils) * sin_tts
    cdo = np.outer(cos_ttli, np.ones(nlazi)) * cos_tto + np.outer(sin_ttli, cos_philo) * sin_tto

    fs = cds / cos_tts
    fo = cdo / cos_tto

    # Use Fortran order ('F') to match MATLAB's column-major reshape behavior
    # This ensures geometric factors align correctly with LIDF weights
    absfs = np.abs(fs).flatten('F')
    absfo = np.abs(fo).flatten('F')
    fsfo = (fs * fo).flatten('F')
    absfsfo = np.abs(fsfo)
    foctl = (fo * np.outer(cos_ttli, np.ones(nlazi))).flatten('F')
    fsctl = (fs * np.outer(cos_ttli, np.ones(nlazi))).flatten('F')
    ctl2 = np.outer(cos_ttli ** 2, np.ones(nlazi)).flatten('F')

    # Fluorescence matrices
    Mb = leafopt.Mb
    Mf = leafopt.Mf
    Mplu = 0.5 * (Mb + Mf)
    Mmin = 0.5 * (Mb - Mf)

    # Energy to photon conversion
    wlF_m = wlF_calc * 1e-9
    wlE_m = wlE_calc * 1e-9
    ep = CONSTANTS.A * ephoton(wlF_m)

    # Calculate fluorescence emission for each layer
    for j in range(nl):
        # Convert incoming energy to photons at excitation wavelengths
        Emin_phot = e2phot(wlE_m, Eminf_[:, j])
        Eplu_phot = e2phot(wlE_m, Epluf_[:, j])
        Esun_phot = e2phot(wlE_m, Esunf_)

        # Get fluorescence matrices for this layer (if layer-dependent)
        if Mplu.ndim == 3:
            Mplu_j = Mplu[:, :, j]
            Mmin_j = Mmin[:, :, j]
        else:
            Mplu_j = Mplu
            Mmin_j = Mmin

        # Ensure proper dimensions for matrix multiplication
        n_common = min(Mplu_j.shape[1], len(Emin_phot))
        Mplu_j = Mplu_j[:, :n_common]
        Mmin_j = Mmin_j[:, :n_common]
        Emin_phot = Emin_phot[:n_common]
        Eplu_phot = Eplu_phot[:n_common]
        Esun_phot = Esun_phot[:n_common]

        MpluEmin[:, j] = ep * (Mplu_j @ Emin_phot)
        MpluEplu[:, j] = ep * (Mplu_j @ Eplu_phot)
        MminEmin[:, j] = ep * (Mmin_j @ Emin_phot)
        MminEplu[:, j] = ep * (Mmin_j @ Eplu_phot)
        MpluEsun[:, j] = ep * (Mplu_j @ Esun_phot)
        MminEsun[:, j] = ep * (Mmin_j @ Esun_phot)

    laz = 1.0 / 36

    # Handle etau dimensions
    # Use order='F' (Fortran/column-major) to match MATLAB reshape behavior
    # This ensures eta values align correctly with geometric factors (absfs, fsfo, etc.)
    # which are also flattened with order='F'
    if etau.ndim == 1:
        etau_flat = np.tile(etau[np.newaxis, :], (nlori, 1))
    elif etau.ndim == 3:
        etau_flat = etau.reshape((nlori, nl), order='F') if etau.shape == (nlinc, nlazi, nl) else np.tile(etau.flatten(order='F')[:, np.newaxis], (1, nl))
    else:
        etau_flat = np.tile(etau.flatten(order='F')[:, np.newaxis], (1, nl))

    # Ensure etau_flat has correct shape
    if etau_flat.shape[0] != nlori:
        etau_flat = np.tile(etau_flat[:1, :], (nlori, 1))
    if etau_flat.shape[1] != nl:
        etau_flat = np.tile(etau_flat[:, :1], (1, nl))

    # Weight by LIDF
    lidf_laz = np.tile(lidf * laz, nlazi)
    etau_lidf = etau_flat * lidf_laz[:, np.newaxis]

    # Handle etah dimensions (same order='F' logic as etau)
    if etah.ndim == 1:
        etah_flat = np.tile(etah[np.newaxis, :], (nlori, 1))
    elif etah.ndim == 3:
        etah_flat = etah.reshape((nlori, nl), order='F') if etah.shape == (nlinc, nlazi, nl) else np.tile(etah.flatten(order='F')[:, np.newaxis], (1, nl))
    else:
        etah_flat = np.tile(etah.flatten(order='F')[:, np.newaxis], (1, nl))

    if etah_flat.shape[0] != nlori:
        etah_flat = np.tile(etah_flat[:1, :], (nlori, 1))
    if etah_flat.shape[1] != nl:
        etah_flat = np.tile(etah_flat[:, :1], (1, nl))

    etah_lidf = etah_flat * lidf_laz[:, np.newaxis]

    # Calculate weighted emission factors
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

    # Emitted fluorescence components (matching MATLAB equations)
    wfEs = MpluEsun * sum_etau_absfsfo + MminEsun * sum_etau_fsfo
    sfEs = MpluEsun * sum_etau_absfs - MminEsun * sum_etau_fsctl
    sbEs = MpluEsun * sum_etau_absfs + MminEsun * sum_etau_fsctl

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

    # Total emitted fluorescence
    piLs = wfEs + vfEplu_u + vbEmin_u  # Sunlit
    piLd = vbEmin_h + vfEplu_h  # Shaded
    Fsmin = sfEs + sigfEmin_u + sigbEplu_u  # Downward from sunlit
    Fsplu = sbEs + sigbEmin_u + sigfEplu_u  # Upward from sunlit
    Fdmin = sigfEmin_h + sigbEplu_h  # Downward from shaded
    Fdplu = sigbEmin_h + sigfEplu_h  # Upward from shaded

    # Layer emission weighted by sunlit/shaded fraction
    Femmin = iLAI * (Fsmin * Qs + Fdmin * (1 - Qs))
    Femplu = iLAI * (Fsplu * Qs + Fdplu * (1 - Qs))

    # Adding method (from bottom to top)
    U = np.zeros((nl + 1, nf))
    Y = np.zeros((nl, nf))

    for j in range(nl - 1, -1, -1):
        denom = 1 - rho_dd[j, :] * R_dd[j + 1, :]
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        Y[j, :] = (rho_dd[j, :] * U[j + 1, :] + Femmin[:, j]) / denom
        U[j, :] = tau_dd[j, :] * (R_dd[j + 1, :] * Y[j, :] + U[j + 1, :]) + Femplu[:, j]

    # Propagate through canopy (from top to bottom)
    Fmin_ = np.zeros((nl + 1, nf))
    Fplu_ = np.zeros((nl + 1, nf))

    for j in range(nl):
        Fmin_[j + 1, :] = Xdd[j, :] * Fmin_[j, :] + Y[j, :]
        Fplu_[j, :] = R_dd[j, :] * Fmin_[j, :] + U[j, :]

    # Calculate observed radiance components
    # Note: vb, vf have shape (nl, nf), Fmin_, Fplu_ have shape (nl+1, nf)
    # piLs, piLd have shape (nf, nl) from the emission calculations
    layers = np.arange(nl)
    piLo1 = iLAI * (piLs @ Pso[:nl])  # Direct sunlit
    piLo2 = iLAI * (piLd @ (Po[:nl] - Pso[:nl]))  # Direct shaded
    # Scattered: (vb * Fmin + vf * Fplu) is (nl, nf), transpose to (nf, nl), then @ Po[:nl]
    piLo3 = iLAI * ((vb * Fmin_[:nl, :] + vf * Fplu_[:nl, :]).T @ Po[:nl])  # Scattered
    piLo4 = rs * Fmin_[nl, :] * Po[nl]  # Soil reflected

    piLtot = piLo1 + piLo2 + piLo3 + piLo4
    LoF_ = piLtot / np.pi
    Fhem_ = Fplu_[0, :]

    # Interpolate to output wavelength grid
    try:
        f_LoF = interp1d(wlF_calc, LoF_, kind='cubic', fill_value='extrapolate')
        f_Fhem = interp1d(wlF_calc, Fhem_, kind='cubic', fill_value='extrapolate')
        f_piLo1 = interp1d(wlF_calc, piLo1 / np.pi, kind='cubic', fill_value='extrapolate')
        f_piLo2 = interp1d(wlF_calc, piLo2 / np.pi, kind='cubic', fill_value='extrapolate')
        f_piLo3 = interp1d(wlF_calc, piLo3 / np.pi, kind='cubic', fill_value='extrapolate')
        f_piLo4 = interp1d(wlF_calc, piLo4 / np.pi, kind='cubic', fill_value='extrapolate')
        f_Fem = interp1d(wlF_calc, np.sum(Femmin + Femplu, axis=1), kind='cubic', fill_value='extrapolate')

        output.LoF_ = f_LoF(spectral.wlF)
        output.EoutF_ = f_Fhem(spectral.wlF)
        output.LoF_sunlit = f_piLo1(spectral.wlF)
        output.LoF_shaded = f_piLo2(spectral.wlF)
        output.LoF_scattered = f_piLo3(spectral.wlF)
        output.LoF_soil = f_piLo4(spectral.wlF)
        output.Femleaves_ = f_Fem(spectral.wlF)
    except Exception:
        # Fallback to linear interpolation
        output.LoF_ = np.interp(spectral.wlF, wlF_calc, LoF_)
        output.EoutF_ = np.interp(spectral.wlF, wlF_calc, Fhem_)
        output.LoF_sunlit = np.interp(spectral.wlF, wlF_calc, piLo1 / np.pi)
        output.LoF_shaded = np.interp(spectral.wlF, wlF_calc, piLo2 / np.pi)
        output.LoF_scattered = np.interp(spectral.wlF, wlF_calc, piLo3 / np.pi)
        output.LoF_soil = np.interp(spectral.wlF, wlF_calc, piLo4 / np.pi)
        output.Femleaves_ = np.interp(spectral.wlF, wlF_calc, np.sum(Femmin + Femplu, axis=1))

    # Integrated values
    output.EoutF = 0.001 * sint(Fhem_, wlF_calc)
    output.LoutF = 0.001 * sint(LoF_, wlF_calc)

    # Peak detection - matching MATLAB RTMf.m indices
    # MATLAB: spectral.wlF = 640:1:850 (indices 1-211, 1-based)
    # Python: spectral.wlF = 640:850 (indices 0-210, 0-based)
    wlF_out = spectral.wlF
    wlF_start = int(wlF_out[0])  # Should be 640

    # F685: MATLAB max(LoF_(1:55)) → indices 0-54 (0-based) → 640-694 nm
    iwl685 = np.argmax(output.LoF_[0:55])
    output.F685 = output.LoF_[iwl685]
    output.wl685 = wlF_out[iwl685]
    # If peak is at edge (index 54), set to NaN (matching MATLAB behavior)
    if iwl685 == 54:
        output.F685 = np.nan
        output.wl685 = np.nan

    # F740: MATLAB max(LoF_(70:end)) → indices 69:end (0-based) → 709-850 nm
    iwl740_rel = np.argmax(output.LoF_[69:])
    iwl740 = iwl740_rel + 69
    output.F740 = output.LoF_[iwl740]
    output.wl740 = wlF_out[iwl740]

    # F684: MATLAB LoF_(685-640) = LoF_(45) → index 44 (0-based)
    idx_684 = 685 - wlF_start  # = 45 for wlF_start=640, but need 0-based: 44
    idx_684 = idx_684 - 1  # Convert to 0-based
    output.F684 = output.LoF_[idx_684] if idx_684 < len(output.LoF_) else 0.0

    # F761: MATLAB LoF_(762-640) = LoF_(122) → index 121 (0-based)
    idx_761 = 762 - wlF_start  # = 122 for wlF_start=640, but need 0-based: 121
    idx_761 = idx_761 - 1  # Convert to 0-based
    output.F761 = output.LoF_[idx_761] if idx_761 < len(output.LoF_) else 0.0

    return output


def interpolate_to_scope_wl(
    wlF: np.ndarray,
    LoF_: np.ndarray,
    wlS: np.ndarray,
) -> np.ndarray:
    """Interpolate fluorescence spectrum to SCOPE wavelength grid.

    Args:
        wlF: Fluorescence wavelengths [nm]
        LoF_: Fluorescence radiance at wlF
        wlS: Target SCOPE wavelengths [nm]

    Returns:
        Interpolated fluorescence on wlS grid
    """
    f = interp1d(wlF, LoF_, kind='cubic', bounds_error=False, fill_value=0.0)
    LoF_interp = f(wlS)
    LoF_interp[wlS < wlF[0]] = 0.0
    LoF_interp[wlS > wlF[-1]] = 0.0
    return LoF_interp
