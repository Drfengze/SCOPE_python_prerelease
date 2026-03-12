"""Output file writing for SCOPE.

This module handles writing simulation outputs to files.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Union

import numpy as np


@dataclass
class OutputFiles:
    """Container for output file handles.
    
    Attributes:
        dir: Output directory path
        binary_files: Dictionary of binary file handles
        text_files: Dictionary of text file handles
    """
    dir: Path
    binary_files: Dict[str, BinaryIO] = field(default_factory=dict)
    text_files: Dict[str, TextIO] = field(default_factory=dict)
    
    def close_all(self):
        """Close all open file handles."""
        for f in self.binary_files.values():
            if not f.closed:
                f.close()
        for f in self.text_files.values():
            if not f.closed:
                f.close()


def create_output_files(
    base_dir: Union[str, Path],
    simulation_name: str,
    options: Optional[Any] = None,
    spectral: Optional[Any] = None,
) -> OutputFiles:
    """Create output directory and files for simulation.
    
    Args:
        base_dir: Base output directory
        simulation_name: Name of simulation (used in directory name)
        options: Simulation options
        spectral: Spectral band definitions
        
    Returns:
        OutputFiles with open file handles
    """
    base_dir = Path(base_dir)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    output_dir = base_dir / f"{simulation_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Parameters subdirectory
    params_dir = output_dir / "Parameters"
    params_dir.mkdir(exist_ok=True)
    
    output_files = OutputFiles(dir=output_dir)
    
    # Define output files
    binary_files = [
        "vegetation.bin",
        "fluxes.bin",
        "radiation.bin",
        "resistances.bin",
        "aPAR.bin",
    ]
    
    # Add optional fluorescence files
    if options is not None and getattr(options, "calc_fluor", False):
        binary_files.extend([
            "fluorescence.bin",
            "fluorescence_scalars.bin",
        ])
    
    # Add optional spectral files
    if options is not None and getattr(options, "save_spectral", False):
        binary_files.extend([
            "reflectance.bin",
            "radiance.bin",
        ])
    
    # Open binary files
    for fname in binary_files:
        fpath = output_dir / fname
        output_files.binary_files[fname] = open(fpath, "wb")
    
    # Save wavelength grids if spectral provided
    if spectral is not None:
        wl_file = output_dir / "wavelengths.txt"
        with open(wl_file, "w") as f:
            f.write("# Wavelength grids for SCOPE output\n")
            f.write(f"# wlS: Full spectrum ({len(spectral.wlS)} bands)\n")
            f.write(f"# wlP: PROSPECT ({len(spectral.wlP)} bands)\n")
            f.write(f"# wlF: Fluorescence ({len(spectral.wlF)} bands)\n")
    
    return output_files


class OutputWriter:
    """Writer for SCOPE simulation outputs.
    
    Handles both binary and text output formats with support for
    time series and spectral data.
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        simulation_name: str = "simulation",
    ):
        """Initialize output writer.
        
        Args:
            output_dir: Directory for output files
            simulation_name: Name of simulation
        """
        self.output_dir = Path(output_dir)
        self.simulation_name = simulation_name
        self._files: Optional[OutputFiles] = None
        self._n_col: Dict[str, int] = {}
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def open(self, options: Optional[Any] = None, spectral: Optional[Any] = None):
        """Open output files.
        
        Args:
            options: Simulation options
            spectral: Spectral band definitions
        """
        self._files = create_output_files(
            self.output_dir,
            self.simulation_name,
            options,
            spectral,
        )
    
    def close(self):
        """Close all output files."""
        if self._files is not None:
            self._files.close_all()
            self._files = None
    
    @property
    def dir(self) -> Path:
        """Get output directory path."""
        if self._files is None:
            raise RuntimeError("Output files not open")
        return self._files.dir
    
    def write_binary(
        self,
        filename: str,
        data: np.ndarray,
        timestep: int = 0,
    ):
        """Write data to binary file.
        
        Args:
            filename: Name of output file
            data: Data array to write
            timestep: Time step index
        """
        if self._files is None:
            raise RuntimeError("Output files not open")
        
        if filename not in self._files.binary_files:
            # Open new file
            fpath = self._files.dir / filename
            self._files.binary_files[filename] = open(fpath, "wb")
        
        f = self._files.binary_files[filename]
        
        # Flatten and write
        data_flat = np.asarray(data).flatten().astype(np.float64)
        data_flat.tofile(f)
        
        # Track columns
        self._n_col[filename] = len(data_flat)
    
    def write_fluxes(
        self,
        timestep: int,
        doy: float,
        Rntot: float,
        lEtot: float,
        Htot: float,
        Rnctot: float,
        lEctot: float,
        Hctot: float,
        Actot: float,
        Rnstot: float,
        lEstot: float,
        Hstot: float,
        Gtot: float,
        **kwargs
    ):
        """Write energy balance fluxes.
        
        Args:
            timestep: Time step index
            doy: Day of year
            Rntot: Total net radiation [W m-2]
            lEtot: Total latent heat [W m-2]
            Htot: Total sensible heat [W m-2]
            Rnctot: Canopy net radiation [W m-2]
            lEctot: Canopy latent heat [W m-2]
            Hctot: Canopy sensible heat [W m-2]
            Actot: Total assimilation [umol m-2 s-1]
            Rnstot: Soil net radiation [W m-2]
            lEstot: Soil latent heat [W m-2]
            Hstot: Soil sensible heat [W m-2]
            Gtot: Soil heat flux [W m-2]
            **kwargs: Additional flux variables
        """
        data = np.array([
            doy,
            Rntot, lEtot, Htot,
            Rnctot, lEctot, Hctot, Actot,
            Rnstot, lEstot, Hstot, Gtot,
        ])
        
        self.write_binary("fluxes.bin", data, timestep)
    
    def write_vegetation(
        self,
        timestep: int,
        Actot: float,
        Ja: float,
        ENPQ: float,
        PNPQ: float,
        eta: float,
        qE: float,
        Fs: float,
        LST: float,
        emis: float,
        GPP: float,
        **kwargs
    ):
        """Write vegetation state variables.
        
        Args:
            timestep: Time step index
            Actot: Total assimilation [umol m-2 s-1]
            Ja: Electron transport rate [umol m-2 s-1]
            ENPQ: Energy-normalized NPQ
            PNPQ: Puddle-normalized NPQ
            eta: Fluorescence efficiency
            qE: Energy-dependent quenching
            Fs: Steady-state fluorescence
            LST: Land surface temperature [C]
            emis: Surface emissivity
            GPP: Gross primary production [umol m-2 s-1]
            **kwargs: Additional vegetation variables
        """
        data = np.array([Actot, Ja, ENPQ, PNPQ, eta, qE, Fs, LST, emis, GPP])
        self.write_binary("vegetation.bin", data, timestep)
    
    def write_radiation(
        self,
        timestep: int,
        Rin: float,
        Rli: float,
        Rout: float,
        PAR: float,
        **kwargs
    ):
        """Write radiation variables.
        
        Args:
            timestep: Time step index
            Rin: Incoming shortwave [W m-2]
            Rli: Incoming longwave [W m-2]
            Rout: Outgoing shortwave [W m-2]
            PAR: Photosynthetically active radiation [umol m-2 s-1]
            **kwargs: Additional radiation variables
        """
        data = np.array([Rin, Rli, Rout, PAR])
        self.write_binary("radiation.bin", data, timestep)
    
    def write_fluorescence(
        self,
        timestep: int,
        F685: float,
        F740: float,
        wl685: float,
        wl740: float,
        F684: float,
        F761: float,
        LoF_: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Write fluorescence outputs.
        
        Args:
            timestep: Time step index
            F685: Fluorescence at 685 nm peak [W m-2 sr-1]
            F740: Fluorescence at 740 nm peak [W m-2 sr-1]
            wl685: Wavelength of 685 nm peak [nm]
            wl740: Wavelength of 740 nm peak [nm]
            F684: Fluorescence at 684 nm [W m-2 sr-1]
            F761: Fluorescence at 761 nm [W m-2 sr-1]
            LoF_: Full fluorescence spectrum [optional]
            **kwargs: Additional fluorescence variables
        """
        # Scalars
        scalars = np.array([F685, F740, wl685, wl740, F684, F761])
        self.write_binary("fluorescence_scalars.bin", scalars, timestep)
        
        # Full spectrum if provided
        if LoF_ is not None:
            self.write_binary("fluorescence.bin", LoF_, timestep)
    
    def write_resistances(
        self,
        timestep: int,
        raa: float,
        raws: float,
        rss: float,
        ustar: float,
        **kwargs
    ):
        """Write aerodynamic resistances.
        
        Args:
            timestep: Time step index
            raa: Aerodynamic resistance [s m-1]
            raws: Soil boundary layer resistance [s m-1]
            rss: Soil surface resistance [s m-1]
            ustar: Friction velocity [m s-1]
            **kwargs: Additional resistance variables
        """
        data = np.array([raa, raws, rss, ustar])
        self.write_binary("resistances.bin", data, timestep)
    
    def write_spectral(
        self,
        timestep: int,
        refl: np.ndarray,
        Lo: Optional[np.ndarray] = None,
    ):
        """Write spectral outputs.
        
        Args:
            timestep: Time step index
            refl: Reflectance spectrum
            Lo: Outgoing radiance spectrum [optional]
        """
        self.write_binary("reflectance.bin", refl, timestep)
        
        if Lo is not None:
            self.write_binary("radiance.bin", Lo, timestep)
    
    def write_csv(
        self,
        filename: str,
        data: Dict[str, Any],
        mode: str = "a",
    ):
        """Write data to CSV file.
        
        Args:
            filename: Name of output file
            data: Dictionary of column names to values
            mode: File mode ("w" for write, "a" for append)
        """
        if self._files is None:
            raise RuntimeError("Output files not open")
        
        fpath = self._files.dir / filename
        write_header = (mode == "w") or not fpath.exists()
        
        with open(fpath, mode) as f:
            if write_header:
                f.write(",".join(data.keys()) + "\n")
            
            values = [str(v) for v in data.values()]
            f.write(",".join(values) + "\n")
