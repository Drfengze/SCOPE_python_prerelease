"""Radiative transfer models for SCOPE.

This module contains the core radiative transfer models:
- Fluspect: Leaf optical properties (PROSPECT with fluorescence)
- BSM: Brightness-Shape-Moisture soil reflectance model
- RTMo: Canopy optical radiative transfer (SAIL-based)
- RTMf: Fluorescence radiative transfer
- RTMt: Thermal radiative transfer
- RTMz: Xanthophyll/PRI effects
"""

from .bsm import BSMParameters, BSMSpectra, bsm, bsm_from_soil, tav
from .fluspect import LeafOpticalOutput, OptiPar, calctav, fluspect, prospect
from .rtmo import (
    GapProbabilities,
    RadiativeTransferOutput,
    calc_fluxprofile,
    calc_reflectances,
    pso_function,
    rtmo,
    volscat,
)
from .rtmf import FluorescenceOutput, rtmf, interpolate_to_scope_wl
from .rtmt import ThermalOutput, rtmt_sb, rtmt_planck, stefan_boltzmann, brightness_temperature
from .rtmz import XanthophyllOutput, rtmz, Kn2Cx, calculate_pri

__all__ = [
    # BSM
    "bsm",
    "bsm_from_soil",
    "BSMParameters",
    "BSMSpectra",
    "tav",
    # Fluspect
    "fluspect",
    "prospect",
    "calctav",
    "OptiPar",
    "LeafOpticalOutput",
    # RTMo
    "rtmo",
    "volscat",
    "pso_function",
    "calc_reflectances",
    "calc_fluxprofile",
    "GapProbabilities",
    "RadiativeTransferOutput",
    # RTMf
    "rtmf",
    "FluorescenceOutput",
    "interpolate_to_scope_wl",
    # RTMt
    "rtmt_sb",
    "rtmt_planck",
    "ThermalOutput",
    "stefan_boltzmann",
    "brightness_temperature",
    # RTMz
    "rtmz",
    "XanthophyllOutput",
    "Kn2Cx",
    "calculate_pri",
]
