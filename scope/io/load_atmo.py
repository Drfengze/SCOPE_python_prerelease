"""Load atmospheric data from MODTRAN files.

Translated from: src/IO/load_atmo.m and supporting functions
"""

import os
import numpy as np
from typing import Dict, Optional, Tuple
from functools import lru_cache

# Module-level cache for atmospheric data
_atmo_cache = {}


def _get_cache_key(atmfile: str, nreg: int, start_tuple: tuple, end_tuple: tuple, res_tuple: tuple) -> str:
    """Create a hashable cache key from file and spectral parameters."""
    return f"{atmfile}_{nreg}_{start_tuple}_{end_tuple}_{res_tuple}"


def aggreg(atmfile: str, spectral) -> np.ndarray:
    """Aggregate MODTRAN data over SCOPE bands by averaging.

    Translated from: src/supporting/aggreg.m

    Args:
        atmfile: Path to .atm file with MODTRAN data
        spectral: Spectral band definitions (needs SCOPEspec info)

    Returns:
        M: Array of shape (nwl, 6) with aggregated MODTRAN data
           Column 0: t1 - Eso*cos(tts)/pi (top-of-atmosphere irradiance)
           Column 1: t3 - rdd (diffuse-diffuse reflectance)
           Column 2: t4 - tss (direct-direct transmittance)
           Column 3: t5 - tsd (direct-diffuse transmittance)
           Column 4: t12 - tssrdd (combined transmittance)
           Column 5: t16 - La(b) (atmospheric path radiance)
    """
    # Read .atm file (skip first 2 header lines)
    with open(atmfile, 'r') as f:
        lines = f.readlines()

    # Find data start (skip header lines)
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith(('WN', '#', 'T', ' ')):
            try:
                float(line.split()[0])
                data_start = i
                break
            except:
                continue

    # Parse data
    data_lines = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) >= 18:
            try:
                values = [float(p) for p in parts]
                data_lines.append(values)
            except:
                continue

    data = np.array(data_lines)
    if len(data) == 0:
        raise ValueError(f"No valid data found in {atmfile}")

    wlM = data[:, 1]  # Wavelength in nm (column 2)
    T = data[:, 2:20]  # Columns 3-20 (18 columns of transmission data)

    # Extract 6 relevant columns:
    # T indices (0-based): 0=toasun, 2=rdd, 3=tss, 4=tsd, 11=tssrdd, 15=Lab
    # These correspond to MATLAB's T(:,1), T(:,3), T(:,4), T(:,5), T(:,12), T(:,16)
    U = np.column_stack([
        T[:, 0],   # t1: toasun (Eso*cos(tts)/pi)
        T[:, 2],   # t3: rdd (diffuse-diffuse reflectance)
        T[:, 3],   # t4: tss (direct-direct transmittance)
        T[:, 4],   # t5: tsd (direct-diffuse transmittance)
        T[:, 11],  # t12: tssrdd
        T[:, 15],  # t16: Lab (atmospheric path radiance)
    ])

    nwM = len(wlM)

    # Get spectral region definitions
    # SCOPEspec has: nreg, start, end, res
    nreg = spectral.nreg
    streg = np.array(spectral.start)
    enreg = np.array(spectral.end)
    width = np.array(spectral.res)

    # Number of bands in each region
    nwreg = ((enreg - streg) / width + 1).astype(int)

    # Offset for each region
    off = np.zeros(nreg, dtype=int)
    for i in range(1, nreg):
        off[i] = off[i-1] + nwreg[i-1]

    nwS = int(np.sum(nwreg))
    n = np.zeros(nwS)  # Count of MODTRAN data contributing to each band
    S = np.zeros((nwS, 6))  # Accumulated sums

    # Aggregate MODTRAN data to SCOPE bands
    for iwl in range(nwM):
        w = wlM[iwl]
        for r in range(nreg):
            j = int(round((w - streg[r]) / width[r])) + 1
            if j > 0 and j <= nwreg[r]:
                k = j - 1 + off[r]  # SCOPE band index (0-based)
                if k < nwS:
                    S[k, :] += U[iwl, :]
                    n[k] += 1

    # Calculate averages
    M = np.zeros((nwS, 6))
    for i in range(6):
        valid = n > 0
        M[valid, i] = S[valid, i] / n[valid]

    return M


def load_atmo(atmfile: str, spectral, use_cache: bool = True) -> Dict:
    """Load atmospheric data from file with caching.

    Translated from: src/IO/load_atmo.m

    Args:
        atmfile: Path to atmospheric file (.atm or .csv)
        spectral: Spectral band definitions
        use_cache: If True, cache and reuse loaded data (default True)

    Returns:
        Dictionary with atmospheric data:
        - If .atm file: 'M' array with MODTRAN data
        - If .csv file: 'Esun_' and 'Esky_' arrays
    """
    global _atmo_cache

    if not os.path.exists(atmfile):
        raise FileNotFoundError(f"Atmospheric file '{atmfile}' does not exist")

    _, ext = os.path.splitext(atmfile)

    if ext == '.atm':
        # Create cache key from file path and spectral parameters
        if use_cache:
            cache_key = _get_cache_key(
                atmfile,
                spectral.nreg,
                tuple(spectral.start),
                tuple(spectral.end),
                tuple(spectral.res)
            )
            if cache_key in _atmo_cache:
                return _atmo_cache[cache_key]

        # MODTRAN format - aggregate to SCOPE bands
        M = aggreg(atmfile, spectral)
        result = {'M': M}

        if use_cache:
            _atmo_cache[cache_key] = result

        return result
    else:
        # Simple two-column format (Esun_, Esky_)
        data = np.loadtxt(atmfile, delimiter=',')
        if data.ndim == 1:
            # Single column - assume it's wavelength-indexed
            raise ValueError("Expected two columns (Esun_, Esky_) in atmospheric file")
        return {
            'Esun_': data[:, 0],
            'Esky_': data[:, 1],
        }


def clear_atmo_cache():
    """Clear the atmospheric data cache."""
    global _atmo_cache
    _atmo_cache = {}


def calc_TOC_irradiance(
    atmo: Dict,
    meteo,
    rdd: np.ndarray,
    rsd: np.ndarray,
    wl: np.ndarray,
    Ta: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate top-of-canopy irradiance from atmospheric data.

    Translated from: RTMo.m/calcTOCirr nested function

    This function computes spectral irradiance using MODTRAN atmospheric
    transfer data, including multiple scattering between atmosphere and surface.

    Args:
        atmo: Atmospheric data dictionary (from load_atmo)
        meteo: Meteorological data (Rin, Rli, Ta)
        rdd: Canopy diffuse-diffuse reflectance [nwl]
        rsd: Canopy direct-diffuse reflectance [nwl]
        wl: Wavelength array [nm]
        Ta: Air temperature [C] for thermal calculation

    Returns:
        Tuple of (Esun_, Esky_) spectral irradiance arrays [mW/m²/nm]
    """
    from ..supporting.physics import planck
    from ..supporting.integration import sint

    nwl = len(wl)

    # Thermal emission from atmosphere
    Ls = planck(wl, Ta + 273.15)
    Fd = np.zeros(nwl)  # Assume Fd of surroundings = 0

    if 'M' in atmo:
        # MODTRAN data available - use full atmospheric RT
        M = atmo['M']

        t1 = M[:, 0]   # Eso*cos(tts)/pi - TOA irradiance
        t3 = M[:, 1]   # rdd - atmospheric diffuse reflectance
        t4 = M[:, 2]   # tss - direct transmittance
        t5 = M[:, 3]   # tsd - direct-diffuse transmittance
        t12 = M[:, 4]  # tssrdd - combined transmittance
        t16 = M[:, 5]  # Lab - atmospheric path radiance

        # Compute irradiances (matching MATLAB exactly)
        # Direct solar: pi * TOA_irradiance * direct_transmittance
        Esun_ = np.maximum(1e-6, np.pi * t1 * t4)

        # Sky diffuse: includes atmospheric scattering, path radiance,
        # and multiple scattering with surface
        # Formula: pi / (1 - rdd_atmo * rdd_surface) *
        #          (TOA * (tsd + tssrdd*rsd) + Fd + (1-rdd)*Ls*rdd_atmo + Lab)
        denominator = 1 - t3 * rdd
        denominator = np.maximum(denominator, 1e-10)  # Prevent division by zero

        Esky_ = np.maximum(1e-6,
            np.pi / denominator * (
                t1 * (t5 + t12 * rsd) +  # Direct scattered + multiple scatter
                Fd +                      # Background (0)
                (1 - rdd) * Ls * t3 +    # Thermal from atmosphere
                t16                       # Path radiance
            )
        )

        # Scale to match measured Rin if provided
        if hasattr(meteo, 'Rin') and meteo.Rin > 0 and meteo.Rin != -999:
            # Optical spectrum (< 3000nm)
            J_o = wl < 3000
            Esunto = 0.001 * sint(Esun_[J_o], wl[J_o])
            Eskyto = 0.001 * sint(Esky_[J_o], wl[J_o])
            Etoto = Esunto + Eskyto

            if Etoto > 0:
                fEsuno = Esun_[J_o] / Etoto
                fEskyo = Esky_[J_o] / Etoto
                Esun_[J_o] = fEsuno * meteo.Rin
                Esky_[J_o] = fEskyo * meteo.Rin

            # Thermal spectrum (>= 3000nm)
            J_t = wl >= 3000
            if np.any(J_t):
                Esuntt = 0.001 * sint(Esun_[J_t], wl[J_t])
                Eskytt = 0.001 * sint(Esky_[J_t], wl[J_t])
                Etott = Esuntt + Eskytt

                if Etott > 0 and hasattr(meteo, 'Rli') and meteo.Rli > 0:
                    fEsunt = Esun_[J_t] / Etott
                    fEskyt = Esky_[J_t] / Etott
                    Esun_[J_t] = fEsunt * meteo.Rli
                    Esky_[J_t] = fEskyt * meteo.Rli
    else:
        # Simple Esun_/Esky_ arrays provided
        Esun_ = atmo.get('Esun_', np.ones(nwl) * 1e-6)
        Esky_ = atmo.get('Esky_', np.ones(nwl) * 1e-6)

    return Esun_, Esky_
