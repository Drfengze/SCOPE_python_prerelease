"""Leaf biochemistry parameters for SCOPE model.

Translated from MATLAB struct definitions in select_input.m and assignvarnames.m.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..constants import TemperatureResponseParams


@dataclass
class LeafBio:
    """Leaf biochemistry and optical parameters.

    This dataclass contains all parameters describing leaf biochemistry,
    optical properties, and photosynthesis characteristics.

    Attributes:
        Cab: Chlorophyll a+b content [µg cm-2]
        Cca: Carotenoid content [µg cm-2]
        Cdm: Dry matter content [g cm-2]
        Cw: Leaf water equivalent layer thickness [cm]
        Cs: Senescent material fraction [-]
        Cant: Anthocyanin content [µg cm-2]
        Cp: Protein content [µg cm-2]
        Cbc: Carbon-based constituents [µg cm-2] (PROSPECT-PRO)
        N: Leaf structure parameter (number of layers) [-]
        rho_thermal: Broadband thermal reflectance [-]
        tau_thermal: Broadband thermal transmittance [-]
        Vcmax25: Maximum carboxylation rate at 25°C [µmol m-2 s-1]
        BallBerrySlope: Ball-Berry stomatal conductance slope [-]
        BallBerry0: Ball-Berry stomatal conductance intercept [mol m-2 s-1]
        Type: Photosynthetic pathway ('C3' or 'C4')
        kV: Vcmax extinction coefficient for canopy [-]
        Rdparam: Ratio of Rd to Vcmax25 [-]
        fqe: Fluorescence quantum efficiency [-] or [2-element array for PSI/PSII]
        Kn0: NPQ parameter Kn0 [-]
        Knalpha: NPQ alpha parameter [-]
        Knbeta: NPQ beta parameter [-]
        kNPQs: Rate constant for sustained thermal dissipation [s-1]
        qLs: Fraction of functional reaction centers [-]
        stressfactor: Stress factor reducing Vcmax [-]
        Tyear: Mean annual temperature [°C]
        beta: Fraction of photons partitioned to PSII [-]
    """

    # Leaf optical/pigment parameters (PROSPECT model)
    Cab: float = 80.0       # Chlorophyll a+b [µg cm-2]
    Cca: float = 20.0       # Carotenoids [µg cm-2]
    Cdm: float = 0.012      # Dry matter [g cm-2]
    Cw: float = 0.009       # Water [cm]
    Cs: float = 0.0         # Senescent material [-]
    Cant: float = 0.0       # Anthocyanins [µg cm-2]
    Cp: float = 0.0         # Protein [µg cm-2]
    Cbc: float = 0.0        # Carbon-based constituents [µg cm-2]
    N: float = 1.4          # Leaf structure parameter [-]

    # Thermal optical properties
    rho_thermal: float = 0.01   # Thermal reflectance [-]
    tau_thermal: float = 0.01   # Thermal transmittance [-]

    # Photosynthesis parameters
    Vcmax25: float = 60.0           # Max carboxylation at 25°C [µmol m-2 s-1]
    BallBerrySlope: float = 8.0     # Ball-Berry slope [-]
    BallBerry0: float = 0.01        # Ball-Berry intercept [mol m-2 s-1]
    Type: Literal["C3", "C4"] = "C3"  # Photosynthetic pathway
    kV: float = 0.6396              # Vcmax extinction coefficient [-]
    RdPerVcmax25: float = 0.015     # Rd/Vcmax ratio [-] (MATLAB name)
    g_m: float = float('inf')       # Mesophyll conductance [mol m-2 s-1 bar-1]

    # Fluorescence parameters
    fqe: float = 0.01               # Fluorescence quantum efficiency [-]

    # Non-photochemical quenching (NPQ) parameters
    Kn0: float = 2.48               # NPQ Kn0 [-]
    Knalpha: float = 2.83           # NPQ alpha [-]
    Knbeta: float = 0.114           # NPQ beta [-]
    kNPQs: float = 0.0              # Sustained NPQ rate [s-1]
    qLs: float = 1.0                # Fraction functional RCs [-]

    # Stress and acclimation
    stressfactor: float = 1.0       # Stress factor [-]
    Tyear: float = 15.0             # Mean annual temperature [°C]
    beta: float = 0.507             # PSII photon fraction (0.507 for C3, 0.4 for C4)

    # Xanthophyll/PRI parameters
    V2Z: float = -999               # Violaxanthin to zeaxanthin conversion factor [-]
                                    # -999 = use standard carotenoid, 0 = violaxanthin, 1 = zeaxanthin

    # Temperature response parameters (TDP structure from MATLAB)
    # If None, default TemperatureResponseParams will be used
    TDP: Optional["TemperatureResponseParams"] = None

    def __post_init__(self) -> None:
        """Validate and compute derived parameters after initialization."""
        # Ensure valid ranges
        if self.Cab < 0:
            raise ValueError(f"Cab must be non-negative, got {self.Cab}")
        if self.N < 1:
            raise ValueError(f"N must be >= 1, got {self.N}")
        if not 0 <= self.rho_thermal <= 1:
            raise ValueError(f"rho_thermal must be in [0,1], got {self.rho_thermal}")
        if not 0 <= self.tau_thermal <= 1:
            raise ValueError(f"tau_thermal must be in [0,1], got {self.tau_thermal}")

        # Set beta based on photosynthetic type if not explicitly set
        if self.Type == "C4" and self.beta == 0.507:
            object.__setattr__(self, "beta", 0.4)

        # Initialize TDP with defaults if not provided
        if self.TDP is None:
            from ..constants import TemperatureResponseParams
            object.__setattr__(self, "TDP", TemperatureResponseParams())

    @property
    def emis(self) -> float:
        """Leaf thermal emissivity [-]."""
        return 1.0 - self.rho_thermal - self.tau_thermal

    @property
    def Rdparam(self) -> float:
        """Alias for RdPerVcmax25 for backwards compatibility."""
        return self.RdPerVcmax25

    @property
    def fqe_array(self) -> NDArray[np.float64]:
        """Fluorescence quantum efficiency as 2-element array [PSI, PSII]."""
        if isinstance(self.fqe, (list, np.ndarray)) and len(self.fqe) == 2:
            return np.asarray(self.fqe)
        # Default: equal contribution from PSI and PSII
        return np.array([self.fqe, self.fqe])

    def copy(self, **kwargs) -> "LeafBio":
        """Create a copy with optionally modified parameters.

        Args:
            **kwargs: Parameters to override in the copy.

        Returns:
            New LeafBio instance with modified parameters.
        """
        from dataclasses import asdict

        params = asdict(self)
        params.update(kwargs)
        return LeafBio(**params)
