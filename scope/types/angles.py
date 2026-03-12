"""Viewing and illumination geometry for SCOPE model.

Translated from MATLAB struct definitions.
"""

from dataclasses import dataclass
from math import cos, sin, radians
from typing import Optional

import numpy as np


@dataclass
class Angles:
    """Sun-observer geometry angles.

    This dataclass contains the angular geometry for illumination
    and observation directions.

    Attributes:
        tts: Solar zenith angle [degrees]
        tto: Observer (viewing) zenith angle [degrees]
        psi: Relative azimuth angle between sun and observer [degrees]
    """

    # Solar angles
    tts: float = 30.0           # Solar zenith [degrees]

    # Observer angles
    tto: float = 0.0            # Observer zenith [degrees]
    psi: float = 90.0           # Relative azimuth [degrees]

    def __post_init__(self) -> None:
        """Validate angles after initialization."""
        if not 0 <= self.tts <= 90:
            raise ValueError(f"Solar zenith angle must be in [0, 90], got {self.tts}")
        if not 0 <= self.tto <= 90:
            raise ValueError(f"Observer zenith angle must be in [0, 90], got {self.tto}")
        if not 0 <= self.psi <= 360:
            raise ValueError(f"Relative azimuth must be in [0, 360], got {self.psi}")

    @property
    def tts_rad(self) -> float:
        """Solar zenith angle in radians."""
        return radians(self.tts)

    @property
    def tto_rad(self) -> float:
        """Observer zenith angle in radians."""
        return radians(self.tto)

    @property
    def psi_rad(self) -> float:
        """Relative azimuth angle in radians."""
        return radians(self.psi)

    @property
    def cos_tts(self) -> float:
        """Cosine of solar zenith angle."""
        return cos(self.tts_rad)

    @property
    def sin_tts(self) -> float:
        """Sine of solar zenith angle."""
        return sin(self.tts_rad)

    @property
    def cos_tto(self) -> float:
        """Cosine of observer zenith angle."""
        return cos(self.tto_rad)

    @property
    def sin_tto(self) -> float:
        """Sine of observer zenith angle."""
        return sin(self.tto_rad)

    @property
    def cos_psi(self) -> float:
        """Cosine of relative azimuth angle."""
        return cos(self.psi_rad)

    @property
    def dso(self) -> float:
        """Sun-observer angular distance (scattering angle cosine).

        This is the angle between sun and observer directions,
        used for hotspot calculations.
        """
        return np.sqrt(
            self.sin_tts**2 + self.sin_tto**2 -
            2 * self.sin_tts * self.sin_tto * self.cos_psi
        )

    def is_hotspot(self, threshold: float = 0.01) -> bool:
        """Check if geometry is near the hotspot (backscatter).

        Args:
            threshold: Angular distance threshold for hotspot [radians]

        Returns:
            True if observer direction is close to solar direction.
        """
        return self.dso < threshold

    def copy(self, **kwargs) -> "Angles":
        """Create a copy with optionally modified parameters.

        Args:
            **kwargs: Parameters to override in the copy.

        Returns:
            New Angles instance with modified parameters.
        """
        from dataclasses import asdict

        params = asdict(self)
        params.update(kwargs)
        return Angles(**params)


@dataclass
class LocationTime:
    """Location and time parameters.

    Attributes:
        LAT: Site latitude [degrees, positive North]
        LON: Site longitude [degrees, positive East]
        timezn: Time zone offset from UTC [hours]
        startDOY: Start day of year [1-365]
        endDOY: End day of year [1-365]
    """

    LAT: float = 52.0           # Latitude [degrees N]
    LON: float = 5.0            # Longitude [degrees E]
    timezn: float = 1.0         # Time zone [hours from UTC]
    startDOY: float = 1.0       # Start day of year
    endDOY: float = 365.0       # End day of year

    def __post_init__(self) -> None:
        """Validate location and time parameters."""
        if not -90 <= self.LAT <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.LAT}")
        if not -180 <= self.LON <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.LON}")
        if not 1 <= self.startDOY <= 366:
            raise ValueError(f"startDOY must be in [1, 366], got {self.startDOY}")
        if not 1 <= self.endDOY <= 366:
            raise ValueError(f"endDOY must be in [1, 366], got {self.endDOY}")
