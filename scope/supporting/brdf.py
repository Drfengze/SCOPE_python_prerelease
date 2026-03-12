"""BRDF (Bidirectional Reflectance Distribution Function) calculations.

Translated from: src/supporting/calc_brdf.m

This module calculates directional reflectance and brightness temperatures
for multiple viewing angles.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class DirectionalOutput:
    """Output from directional (BRDF) calculations.

    Attributes:
        tto: View zenith angles [degrees]
        psi: Relative azimuth angles [degrees]
        refl_: Spectral reflectance at each angle, shape (nwl, nangles)
        rso_: Directional reflectance at each angle, shape (nwl, nangles)
        Lot_: Thermal radiance at each angle (if calc_planck), shape (nwlt, nangles)
        LoF_: Fluorescence radiance at each angle (if calc_fluor), shape (nwlf, nangles)
        BrightnessT: Brightness temperature at each angle [K]
        Eoutte: Thermal emission at each angle [W m-2]
    """
    tto: np.ndarray = field(default_factory=lambda: np.array([]))
    psi: np.ndarray = field(default_factory=lambda: np.array([]))
    refl_: np.ndarray = field(default_factory=lambda: np.array([]))
    rso_: np.ndarray = field(default_factory=lambda: np.array([]))
    Lot_: np.ndarray = field(default_factory=lambda: np.array([]))
    LoF_: np.ndarray = field(default_factory=lambda: np.array([]))
    BrightnessT: np.ndarray = field(default_factory=lambda: np.array([]))
    Eoutte: np.ndarray = field(default_factory=lambda: np.array([]))


def calc_brdf(
    spectral,
    atmo: dict,
    soil,
    leafopt,
    canopy,
    angles,
    meteo,
    options: dict,
    thermal: Optional[dict] = None,
    bcu=None,
    bch=None,
) -> DirectionalOutput:
    """Calculate BRDF (directional reflectance) for multiple viewing angles.

    This function simulates observations from a large number of viewing angles,
    including hotspot sampling around the solar angle.

    Args:
        spectral: Spectral band definitions
        atmo: Atmospheric data (Esun_, Esky_)
        soil: Soil properties
        leafopt: Leaf optical properties
        canopy: Canopy structure
        angles: Base viewing geometry (only tts is used)
        meteo: Meteorological conditions
        options: Simulation options dict
        thermal: Thermal state (Tcu, Tch, Tsu, Tsh) for thermal calculations
        bcu: Biochemical output for sunlit leaves (for fluorescence)
        bch: Biochemical output for shaded leaves (for fluorescence)

    Returns:
        DirectionalOutput with reflectance/radiance at multiple angles
    """
    from ..rtm.rtmo import rtmo
    from ..types import Angles

    output = DirectionalOutput()

    tts = angles.tts

    # Define sampling angles (matching MATLAB calc_brdf.m)
    # Hotspot oversampling: angles near solar direction
    psi_hot = np.array([0, 0, 0, 0, 0, 2, 358])
    tto_hot = np.array([tts, tts + 2, tts + 4, tts - 2, tts - 4, tts, tts])

    # Principal plane sampling: 4 planes at 0°, 90°, 180°, 270°
    psi_plane = np.concatenate([
        np.full(6, 0),
        np.full(6, 180),
        np.full(6, 90),
        np.full(6, 270)
    ])
    tto_plane = np.tile(np.arange(10, 61, 10), 4)

    # Combine all angles
    psi_all = np.concatenate([psi_hot, psi_plane])
    tto_all = np.concatenate([tto_hot, tto_plane])

    # Remove duplicates
    angle_pairs = np.column_stack([psi_all, tto_all])
    _, unique_idx = np.unique(angle_pairs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    psi_unique = psi_all[unique_idx]
    tto_unique = tto_all[unique_idx]
    n_angles = len(unique_idx)

    # Store angle arrays
    output.tto = tto_unique
    output.psi = psi_unique

    # Allocate output arrays
    nwlS = len(spectral.wlS)
    nwlT = len(spectral.wlT)
    nwlF = len(spectral.wlF)

    output.refl_ = np.zeros((nwlS, n_angles))
    output.rso_ = np.zeros((nwlS, n_angles))
    output.Lot_ = np.zeros((nwlT, n_angles))
    output.LoF_ = np.zeros((nwlF, n_angles))
    output.BrightnessT = np.zeros(n_angles)
    output.Eoutte = np.zeros(n_angles)

    # Determine which optional calculations to perform
    calc_planck = options.get('calc_planck', False)
    calc_fluor = options.get('calc_fluor', False)
    calc_xanthophyll = options.get('calc_xanthophyll', False)

    # Loop over angles
    for j in range(n_angles):
        # Create angles for this direction
        dir_angles = Angles(
            tts=tts,
            tto=float(tto_unique[j]),
            psi=float(psi_unique[j]),
        )

        # Run RTMo for this viewing direction
        dir_rad, dir_gap, _ = rtmo(
            spectral=spectral,
            atmo=atmo,
            soil=soil,
            leafopt=leafopt,
            canopy=canopy,
            angles=dir_angles,
            meteo=meteo,
            options=options,
        )

        # Store reflectance results
        output.refl_[:, j] = dir_rad.refl
        output.rso_[:, j] = dir_rad.rso

        # Thermal directional brightness temperatures (if requested and thermal state available)
        if calc_planck and thermal is not None:
            try:
                from ..rtm.rtmt import rtmt_planck

                thermal_out = rtmt_planck(
                    spectral=spectral,
                    rad=dir_rad,
                    soil=soil,
                    leafopt=leafopt,
                    canopy=canopy,
                    gap=dir_gap,
                    Tcu=thermal['Tcu'],
                    Tch=thermal['Tch'],
                    Tsu=thermal['Tsu'],
                    Tsh=thermal['Tsh'],
                )

                # Store thermal + reflected radiance (extract thermal portion)
                Lot_thermal = thermal_out.Lot_[spectral.IwlT] if thermal_out.Lot_.shape[0] > len(spectral.IwlT) else thermal_out.Lot_
                Lot_total = Lot_thermal + dir_rad.Lo_[spectral.IwlT]
                output.Lot_[:, j] = Lot_total
                output.Eoutte[j] = thermal_out.Eoutte
                output.BrightnessT[j] = thermal_out.LST if hasattr(thermal_out, 'LST') else 0.0

            except (ImportError, AttributeError):
                pass

        # Fluorescence (if requested and biochemical outputs available)
        if calc_fluor and bcu is not None and bch is not None:
            try:
                from ..rtm.rtmf import rtmf

                fluor_out = rtmf(
                    spectral=spectral,
                    rad=dir_rad,
                    soil=soil,
                    leafopt=leafopt,
                    canopy=canopy,
                    gap=dir_gap,
                    angles=dir_angles,
                    etau=bcu.eta,
                    etah=bch.eta,
                )

                output.LoF_[:, j] = fluor_out.LoF_

            except (ImportError, AttributeError):
                pass

        # Xanthophyll/PRI (if requested)
        if calc_xanthophyll and bcu is not None and bch is not None:
            try:
                from ..rtm.rtmz import rtmz

                xanth_out = rtmz(
                    spectral=spectral,
                    rad=dir_rad,
                    soil=soil,
                    leafopt=leafopt,
                    canopy=canopy,
                    gap=dir_gap,
                    angles=dir_angles,
                    Knu=bcu.Kn,
                    Knh=bch.Kn,
                )

                # Update radiance and reflectance with xanthophyll effects
                output.refl_[:, j] = xanth_out.refl_mod
                # Note: Lo_ is also modified but we store refl which is derived from it

            except (ImportError, AttributeError):
                pass

    return output


def interpolate_brdf(
    directional: DirectionalOutput,
    tto_target: float,
    psi_target: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate BRDF to a specific viewing angle.

    Args:
        directional: DirectionalOutput from calc_brdf
        tto_target: Target view zenith angle [degrees]
        psi_target: Target relative azimuth angle [degrees]

    Returns:
        Tuple of (refl, rso) interpolated spectra
    """
    from scipy.interpolate import griddata

    # Create coordinate grid
    points = np.column_stack([directional.tto, directional.psi])
    target = np.array([[tto_target, psi_target]])

    # Interpolate reflectance
    refl_interp = np.zeros(directional.refl_.shape[0])
    rso_interp = np.zeros(directional.rso_.shape[0])

    for i in range(directional.refl_.shape[0]):
        refl_interp[i] = griddata(points, directional.refl_[i, :], target, method='linear')[0]
        rso_interp[i] = griddata(points, directional.rso_[i, :], target, method='linear')[0]

    return refl_interp, rso_interp
