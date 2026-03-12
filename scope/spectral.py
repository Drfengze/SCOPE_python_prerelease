"""Spectral band definitions for SCOPE model.

Translated from: src/IO/define_bands.m

SCOPE uses three spectral regions with different resolutions:
1. Optical (PROSPECT): 400-2400 nm at 1 nm resolution (2001 bands)
2. Near thermal: 2500-15000 nm at 100 nm resolution (126 bands)
3. Thermal: 16000-50000 nm at 1000 nm resolution (35 bands)

Total: 2162 spectral bands in wlS
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SpectralRegion:
    """Definition of a single spectral region.

    Attributes:
        start: Start wavelength [nm]
        end: End wavelength [nm]
        resolution: Spectral resolution [nm]
    """

    start: float
    end: float
    resolution: float

    @property
    def wavelengths(self) -> NDArray[np.float64]:
        """Generate wavelength array for this region."""
        return np.arange(self.start, self.end + self.resolution, self.resolution)

    @property
    def n_bands(self) -> int:
        """Number of bands in this region."""
        return len(self.wavelengths)


@dataclass
class SpectralBands:
    """Spectral band definitions for SCOPE.

    This class defines all wavelength arrays and index mappings used in SCOPE.

    Attributes:
        wlS: Full SCOPE spectrum 400-50000 nm (2162 bands)
        wlP: PROSPECT range 400-2400 nm (2001 bands)
        wlO: Optical range (same as wlP)
        wlE: Excitation wavelengths 400-750 nm (351 bands)
        wlF: Fluorescence emission 640-850 nm (211 bands)
        wlT: Thermal range 2500-50000 nm (161 bands)
        wlZ: Xanthophyll/PRI range 500-600 nm (101 bands)
        wlPAR: PAR range 400-700 nm (301 bands)
    """

    # Spectral regions definition (from SCOPEspec in MATLAB)
    _regions: tuple[SpectralRegion, ...] = field(
        default_factory=lambda: (
            SpectralRegion(start=400, end=2400, resolution=1),     # Optical (PROSPECT)
            SpectralRegion(start=2500, end=15000, resolution=100),  # Near thermal
            SpectralRegion(start=16000, end=50000, resolution=1000),  # Thermal
        ),
        repr=False,
    )

    def __post_init__(self) -> None:
        """Initialize all wavelength arrays after dataclass creation."""
        # Generate full SCOPE spectrum by concatenating regions
        wl_parts = [region.wavelengths for region in self._regions]
        self._wlS = np.concatenate(wl_parts)

        # PROSPECT wavelength range (optical, 1 nm resolution)
        self._wlP = np.arange(400, 2401, 1, dtype=np.float64)

        # Excitation wavelengths for fluorescence
        self._wlE = np.arange(400, 751, 1, dtype=np.float64)

        # Fluorescence emission wavelengths
        self._wlF = np.arange(640, 851, 1, dtype=np.float64)

        # Xanthophyll/PRI wavelengths
        self._wlZ = np.arange(500, 601, 1, dtype=np.float64)

        # PAR wavelengths
        self._wlPAR = np.arange(400, 701, 1, dtype=np.float64)

        # Thermal wavelengths (combine near-thermal and thermal)
        wlT1 = np.arange(2500, 15001, 100, dtype=np.float64)
        wlT2 = np.arange(16000, 50001, 1000, dtype=np.float64)
        self._wlT = np.concatenate([wlT1, wlT2])

        # Calculate index arrays for mapping between different wavelength grids
        self._calculate_indices()

    def _calculate_indices(self) -> None:
        """Calculate index arrays for wavelength mappings."""
        # Indices of PROSPECT wavelengths in full spectrum (1-based in MATLAB, 0-based here)
        # wlP covers indices 0-2000 in wlS
        self._IwlP = np.arange(0, len(self._wlP), dtype=np.int64)

        # Indices of thermal wavelengths in full spectrum
        # Thermal starts after PROSPECT (index 2001 onwards)
        self._IwlT = np.arange(len(self._wlP), len(self._wlS), dtype=np.int64)

        # Indices of fluorescence wavelengths in full spectrum
        # wlF: 640-850 nm, which starts at index 240 (640-400=240) in wlS
        iwlf_start = int(self._wlF[0] - self._wlS[0])
        iwlf_end = iwlf_start + len(self._wlF)
        self._IwlF = np.arange(iwlf_start, iwlf_end, dtype=np.int64)

        # Indices of excitation wavelengths in full spectrum
        # wlE: 400-750 nm, starts at index 0
        self._IwlE = np.arange(0, len(self._wlE), dtype=np.int64)

        # Indices of PAR wavelengths in full spectrum
        # wlPAR: 400-700 nm, starts at index 0
        self._IwlPAR = np.arange(0, len(self._wlPAR), dtype=np.int64)

        # Indices of xanthophyll wavelengths in full spectrum
        # wlZ: 500-600 nm, starts at index 100 (500-400=100)
        iwlz_start = int(self._wlZ[0] - self._wlS[0])
        iwlz_end = iwlz_start + len(self._wlZ)
        self._IwlZ = np.arange(iwlz_start, iwlz_end, dtype=np.int64)

    @property
    def wlS(self) -> NDArray[np.float64]:
        """Full SCOPE spectrum 400-50000 nm (2162 bands)."""
        return self._wlS

    @property
    def wlP(self) -> NDArray[np.float64]:
        """PROSPECT wavelength range 400-2400 nm (2001 bands)."""
        return self._wlP

    @property
    def wlO(self) -> NDArray[np.float64]:
        """Optical wavelength range (same as wlP)."""
        return self._wlP

    @property
    def wlE(self) -> NDArray[np.float64]:
        """Excitation wavelengths 400-750 nm (351 bands)."""
        return self._wlE

    @property
    def wlF(self) -> NDArray[np.float64]:
        """Fluorescence emission wavelengths 640-850 nm (211 bands)."""
        return self._wlF

    @property
    def wlT(self) -> NDArray[np.float64]:
        """Thermal wavelengths 2500-50000 nm (161 bands)."""
        return self._wlT

    @property
    def wlZ(self) -> NDArray[np.float64]:
        """Xanthophyll/PRI wavelengths 500-600 nm (101 bands)."""
        return self._wlZ

    @property
    def wlPAR(self) -> NDArray[np.float64]:
        """PAR wavelengths 400-700 nm (301 bands)."""
        return self._wlPAR

    @property
    def IwlP(self) -> NDArray[np.int64]:
        """Indices of PROSPECT wavelengths in wlS."""
        return self._IwlP

    @property
    def IwlT(self) -> NDArray[np.int64]:
        """Indices of thermal wavelengths in wlS."""
        return self._IwlT

    @property
    def IwlF(self) -> NDArray[np.int64]:
        """Indices of fluorescence wavelengths in wlS."""
        return self._IwlF

    @property
    def IwlE(self) -> NDArray[np.int64]:
        """Indices of excitation wavelengths in wlS."""
        return self._IwlE

    @property
    def IwlPAR(self) -> NDArray[np.int64]:
        """Indices of PAR wavelengths in wlS."""
        return self._IwlPAR

    @property
    def IwlZ(self) -> NDArray[np.int64]:
        """Indices of xanthophyll wavelengths in wlS."""
        return self._IwlZ

    @property
    def nwlS(self) -> int:
        """Total number of bands in full spectrum."""
        return len(self._wlS)

    @property
    def nwlP(self) -> int:
        """Number of bands in PROSPECT range."""
        return len(self._wlP)

    @property
    def nwlF(self) -> int:
        """Number of bands in fluorescence range."""
        return len(self._wlF)

    @property
    def nwlE(self) -> int:
        """Number of bands in excitation range."""
        return len(self._wlE)

    @property
    def nwlT(self) -> int:
        """Number of bands in thermal range."""
        return len(self._wlT)

    @property
    def nwlPAR(self) -> int:
        """Number of bands in PAR range."""
        return len(self._wlPAR)

    # SCOPEspec properties for MODTRAN aggregation
    @property
    def nreg(self) -> int:
        """Number of spectral regions."""
        return len(self._regions)

    @property
    def start(self) -> list:
        """Start wavelengths of each region [nm]."""
        return [r.start for r in self._regions]

    @property
    def end(self) -> list:
        """End wavelengths of each region [nm]."""
        return [r.end for r in self._regions]

    @property
    def res(self) -> list:
        """Resolution of each region [nm]."""
        return [r.resolution for r in self._regions]

    def get_wavelength_indices(
        self, wl_start: float, wl_end: float, in_array: str = "wlS"
    ) -> NDArray[np.int64]:
        """Get indices for a wavelength range in the specified array.

        Args:
            wl_start: Start wavelength [nm]
            wl_end: End wavelength [nm]
            in_array: Name of wavelength array to search ('wlS', 'wlP', 'wlF', etc.)

        Returns:
            Array of indices where wavelengths fall within the specified range.
        """
        wl_array = getattr(self, in_array)
        mask = (wl_array >= wl_start) & (wl_array <= wl_end)
        return np.where(mask)[0]


# Module-level singleton for convenience
SPECTRAL = SpectralBands()
