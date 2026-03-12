"""Radiation output structures for SCOPE model.

Translated from MATLAB struct definitions.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class LeafOptics:
    """Leaf optical properties from Fluspect model.

    Attributes:
        refl: Leaf reflectance spectrum (nwlP,)
        tran: Leaf transmittance spectrum (nwlP,)
        kChlrel: Relative chlorophyll absorption (nwlP,)
        Mb: Backward fluorescence matrix (nwlF, nwlE)
        Mf: Forward fluorescence matrix (nwlF, nwlE)
    """

    refl: Optional[NDArray[np.float64]] = None
    tran: Optional[NDArray[np.float64]] = None
    kChlrel: Optional[NDArray[np.float64]] = None
    Mb: Optional[NDArray[np.float64]] = None
    Mf: Optional[NDArray[np.float64]] = None


@dataclass
class CanopyRadiation:
    """Canopy radiative transfer outputs.

    This dataclass contains all outputs from the RTMo canopy
    optical radiative transfer model.

    Attributes:
        Emins: Direct solar irradiance incident on soil (nwlS,)
        Emin: Diffuse downward flux at soil (nwlS,)
        Eplu: Diffuse upward flux at canopy top (nwlS,)
        Lo: Observed radiance at sensor (nwlS,)
        Eout: Total outgoing radiance at canopy top (nwlS,)
        Rnuc: Net radiation at each canopy layer (nlayers,)
        Pnuc: Net PAR at each layer (nlayers,)
        Pnuc_Cab: Net PAR absorbed by chlorophyll (nlayers,)
        fEsuno: Fraction of sunlit leaves at each layer (nlayers,)
        fEshao: Fraction of shaded leaves at each layer (nlayers,)
        Ps: Gap probability (sun direction) (nlayers+1,)
        Po: Gap probability (observer direction) (nlayers+1,)
        Pso: Bidirectional gap probability (nlayers+1,)
        rso: Soil bidirectional reflectance (nwlP,)
        rdo: Soil directional-hemispherical reflectance (nwlP,)
        rsd: Soil hemispherical-directional reflectance (nwlP,)
        rdd: Soil hemispherical-hemispherical reflectance (nwlP,)
    """

    # Spectral fluxes at boundaries
    Emins: Optional[NDArray[np.float64]] = None  # Direct solar at soil
    Emin: Optional[NDArray[np.float64]] = None   # Diffuse downward at soil
    Eplu: Optional[NDArray[np.float64]] = None   # Diffuse upward at top
    Lo: Optional[NDArray[np.float64]] = None     # Observed radiance
    Eout: Optional[NDArray[np.float64]] = None   # Total outgoing

    # Layer-resolved quantities
    Rnuc: Optional[NDArray[np.float64]] = None       # Net radiation per layer
    Pnuc: Optional[NDArray[np.float64]] = None       # Net PAR per layer
    Pnuc_Cab: Optional[NDArray[np.float64]] = None   # PAR absorbed by Chl

    # Sunlit/shaded fractions
    fEsuno: Optional[NDArray[np.float64]] = None     # Sunlit fraction
    fEshao: Optional[NDArray[np.float64]] = None     # Shaded fraction

    # Gap probabilities
    Ps: Optional[NDArray[np.float64]] = None         # Sun direction
    Po: Optional[NDArray[np.float64]] = None         # Observer direction
    Pso: Optional[NDArray[np.float64]] = None        # Bidirectional

    # Canopy reflectance factors (for multi-layer scattering)
    rso: Optional[NDArray[np.float64]] = None
    rdo: Optional[NDArray[np.float64]] = None
    rsd: Optional[NDArray[np.float64]] = None
    rdd: Optional[NDArray[np.float64]] = None

    # Integrated quantities
    PAR: float = 0.0                 # Total PAR [W m-2]
    fPAR: float = 0.0                # Fraction of PAR absorbed [-]
    albedo: float = 0.0              # Shortwave albedo [-]


@dataclass
class FluorescenceOutput:
    """Fluorescence radiative transfer outputs.

    Attributes:
        LoF: Observed fluorescence radiance (nwlF,)
        Fhem: Hemispherical fluorescence flux at canopy top (nwlF,)
        Fem: Total emitted fluorescence (nwlF,)
        F685: Fluorescence at 685 nm [W m-2 sr-1 nm-1]
        F740: Fluorescence at 740 nm [W m-2 sr-1 nm-1]
        Fhem685: Hemispherical fluorescence at 685 nm [W m-2 nm-1]
        Fhem740: Hemispherical fluorescence at 740 nm [W m-2 nm-1]
        eta: Fluorescence efficiency per layer (nlayers,)
    """

    LoF: Optional[NDArray[np.float64]] = None        # Observed SIF
    Fhem: Optional[NDArray[np.float64]] = None       # Hemispherical SIF
    Fem: Optional[NDArray[np.float64]] = None        # Total emitted

    # Key wavelength values
    F685: float = 0.0                # SIF at 685 nm
    F740: float = 0.0                # SIF at 740 nm
    Fhem685: float = 0.0             # Hemispherical at 685 nm
    Fhem740: float = 0.0             # Hemispherical at 740 nm

    # Layer-resolved fluorescence efficiency
    eta: Optional[NDArray[np.float64]] = None


@dataclass
class ThermalOutput:
    """Thermal radiative transfer outputs.

    Attributes:
        Lot: Observed thermal radiance (nwlT,)
        Lot_: Thermal radiance without atmosphere
        Lote: Emitted thermal radiance
        LST: Land surface temperature [K]
        Tc: Canopy temperature [K]
        Ts: Soil surface temperature [K]
    """

    Lot: Optional[NDArray[np.float64]] = None
    Lot_: Optional[NDArray[np.float64]] = None
    Lote: Optional[NDArray[np.float64]] = None

    LST: float = 0.0                 # Land surface temperature [K]
    Tc: float = 0.0                  # Canopy temperature [K]
    Ts: float = 0.0                  # Soil temperature [K]


@dataclass
class Radiation:
    """Combined radiation outputs from all RTM modules.

    This is the main output structure containing results from
    optical, fluorescence, and thermal radiative transfer.

    Attributes:
        optics: Leaf optical properties from Fluspect
        canopy: Canopy optical RTM outputs
        fluorescence: Fluorescence RTM outputs
        thermal: Thermal RTM outputs
    """

    optics: LeafOptics = field(default_factory=LeafOptics)
    canopy: CanopyRadiation = field(default_factory=CanopyRadiation)
    fluorescence: FluorescenceOutput = field(default_factory=FluorescenceOutput)
    thermal: ThermalOutput = field(default_factory=ThermalOutput)
