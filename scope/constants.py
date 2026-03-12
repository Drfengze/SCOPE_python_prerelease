"""Physical constants used in SCOPE model.

Translated from: src/IO/define_constants.m
"""

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class Constants:
    """Physical constants for SCOPE model.

    All values are immutable (frozen dataclass) to prevent accidental modification.

    Attributes:
        A: Avogadro's constant [mol-1]
        h: Planck's constant [J s]
        c: Speed of light in vacuum [m s-1]
        cp: Specific heat capacity of dry air at constant pressure [J kg-1 K-1]
        R: Universal molar gas constant [J mol-1 K-1]
        rhoa: Density of air at sea level [kg m-3]
        g: Gravitational acceleration [m s-2]
        kappa: Von Karman constant [-]
        MH2O: Molecular mass of water [g mol-1]
        Mair: Molecular mass of dry air [g mol-1]
        MCO2: Molecular mass of carbon dioxide [g mol-1]
        sigmaSB: Stefan-Boltzmann constant [W m-2 K-4]
        deg2rad: Conversion factor from degrees to radians [rad deg-1]
        C2K: Offset to convert Celsius to Kelvin [K]
    """

    # Avogadro's constant [mol-1]
    A: float = 6.02214e23

    # Planck's constant [J s]
    h: float = 6.6262e-34

    # Speed of light in vacuum [m s-1]
    c: float = 299792458.0

    # Specific heat capacity of dry air at constant pressure [J kg-1 K-1]
    cp: float = 1004.0

    # Universal molar gas constant [J mol-1 K-1]
    R: float = 8.314

    # Density of air at sea level [kg m-3]
    rhoa: float = 1.2047

    # Gravitational acceleration [m s-2]
    g: float = 9.81

    # Von Karman constant [-]
    kappa: float = 0.4

    # Molecular mass of water [g mol-1]
    MH2O: float = 18.0

    # Molecular mass of dry air [g mol-1]
    Mair: float = 28.96

    # Molecular mass of carbon dioxide [g mol-1]
    MCO2: float = 44.0

    # Stefan-Boltzmann constant [W m-2 K-4]
    sigmaSB: float = 5.67e-8

    # Conversion factor from degrees to radians [rad deg-1]
    deg2rad: float = pi / 180.0

    # Offset to convert Celsius to Kelvin [K]
    C2K: float = 273.15


@dataclass(frozen=True)
class TemperatureResponseParams:
    """Temperature response parameters for biochemical model.

    Translated from: src/IO/define_temp_response_biochem.m
    Based on CLM4 temperature sensitivity parameterization.

    These parameters define the temperature dependence of:
    - Vcmax: Maximum carboxylation rate
    - Jmax: Maximum electron transport rate
    - TPU: Triose phosphate utilization
    - Rd: Dark respiration
    - Kc, Ko: Michaelis-Menten constants for Rubisco
    """

    # Vcmax temperature response
    delHaV: float = 65330.0    # Activation energy [J mol-1]
    delSV: float = 485.0       # Entropy term [J mol-1 K-1]
    delHdV: float = 149250.0   # Deactivation energy [J mol-1]

    # Jmax temperature response
    delHaJ: float = 43540.0    # Activation energy [J mol-1]
    delSJ: float = 495.0       # Entropy term [J mol-1 K-1]
    delHdJ: float = 152040.0   # Deactivation energy [J mol-1]

    # TPU (photorespiration) temperature response
    delHaP: float = 53100.0    # Activation energy [J mol-1]
    delSP: float = 490.0       # Entropy term [J mol-1 K-1]
    delHdP: float = 150650.0   # Deactivation energy [J mol-1]

    # Rd (respiration) temperature response
    delHaR: float = 46390.0    # Activation energy [J mol-1]
    delSR: float = 490.0       # Entropy term [J mol-1 K-1]
    delHdR: float = 150650.0   # Deactivation energy [J mol-1]

    # Rubisco Km for CO2 temperature response
    delHaKc: float = 79430.0   # Activation energy [J mol-1]

    # Rubisco Km for O2 temperature response
    delHaKo: float = 36380.0   # Activation energy [J mol-1]

    # Tau (CO2/O2 specificity) temperature response
    delHaT: float = 37830.0    # Activation energy [J mol-1]

    # Q10 coefficient for exponential temperature response
    Q10: float = 2.0

    # High temperature inhibition parameters
    s1: float = 0.3
    s2: float = 313.15         # 40°C in Kelvin

    # Low temperature inhibition parameters
    s3: float = 0.2
    s4: float = 288.15         # 15°C in Kelvin

    # Extreme high temperature parameters
    s5: float = 1.3
    s6: float = 328.15         # 55°C in Kelvin


# Module-level singleton instances for convenience
CONSTANTS = Constants()
TEMP_RESPONSE = TemperatureResponseParams()
