"""Simulation options for SCOPE model.

Translated from: src/IO/setoptions.csv and related option handling.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Options:
    """Simulation options controlling SCOPE behavior.

    These options control which model components are activated and how
    the simulation is performed.

    Attributes:
        simulation: Simulation mode (0=individual, 1=time series, 2=LUT)
        calc_fluor: Calculate chlorophyll fluorescence
        calc_planck: Use Planck function for thermal (True) or Stefan-Boltzmann (False)
        calc_xanthophyll: Calculate xanthophyll cycle / PRI effects
        calc_directional: Calculate directional reflectance (BRDF)
        calc_vert_profiles: Calculate vertical profiles of photosynthesis
        rt_thermal: Include thermal radiation in RTM
        calc_ebal: Calculate energy balance (False = prescribed leaf temperature)
        soil_heat_method: Soil heat flux method (0=constant fraction, 1=force restore, 2=full model)
        calc_rss_rbs: Calculate soil resistance for evaporation
        apply_T_corr: Apply temperature corrections to biochemistry
        verify: Run verification against stored outputs
        save_headers: Save headers in output files
        Fluorescence_model: Fluorescence model (0=empirical, 1=mechanistic)
        stomatal_model: Stomatal conductance model (0=Ball-Berry, 1=Medlyn)
        apply_vcmax_profile: Apply vertical Vcmax profile in canopy
        soilspectrum: Soil spectrum source (1=file, 2=BSM model)
        mSCOPE: Use mSCOPE multi-layer mode
        save_spectral: Save full spectral outputs
        calc_zo: Calculate roughness length (0=use input, 1=calculate)
    """

    # Simulation mode: 0=individual run, 1=time series, 2=lookup table
    simulation: Literal[0, 1, 2] = 0

    # Calculate chlorophyll fluorescence
    calc_fluor: bool = True

    # Use Planck function for thermal radiation (False = Stefan-Boltzmann)
    calc_planck: bool = True

    # Calculate xanthophyll cycle / PRI effects
    calc_xanthophyll: bool = False

    # Calculate directional reflectance (BRDF)
    calc_directional: bool = False

    # Calculate vertical profiles of photosynthesis
    calc_vert_profiles: bool = False

    # Include thermal radiation in radiative transfer
    rt_thermal: bool = True

    # Calculate energy balance (False = use prescribed leaf temperature)
    calc_ebal: bool = True

    # Soil heat flux calculation method:
    # 0 = constant fraction of Rn
    # 1 = force-restore method
    # 2 = full soil heat transfer model
    soil_heat_method: Literal[0, 1, 2] = 0

    # Calculate soil resistances for evaporation
    calc_rss_rbs: bool = False

    # Apply temperature corrections to photosynthesis parameters
    apply_T_corr: bool = True

    # Run verification mode (compare to stored outputs)
    verify: bool = False

    # Save headers in output CSV files
    save_headers: bool = True

    # Fluorescence model: 0=empirical (constant fqe), 1=mechanistic (Kn-based)
    Fluorescence_model: Literal[0, 1] = 0

    # Stomatal conductance model: 0=Ball-Berry, 1=Medlyn
    stomatal_model: Literal[0, 1] = 0

    # Apply exponential Vcmax profile through canopy
    apply_vcmax_profile: bool = False

    # Soil spectrum source: 1=from file, 2=BSM model
    soilspectrum: Literal[1, 2] = 1

    # Use mSCOPE multi-layer leaf biochemistry
    mSCOPE: bool = False

    # Save full spectral outputs (increases output file size)
    save_spectral: bool = False

    # Calculate roughness length: 0=use input values, 1=calculate from canopy
    calc_zo: Literal[0, 1] = 0

    # Use SCOPE-lite mode (layer-averaged instead of full angular integration)
    # True = lite mode (faster, less memory)
    # False = full mode (13 inclination x 36 azimuth classes)
    lite: bool = True

    # Monin-Obukhov stability correction
    MoninObukhov: bool = True


@dataclass
class FilePaths:
    """File path configuration for SCOPE inputs and outputs.

    Attributes:
        input_dir: Path to input data directory
        output_dir: Path to output directory
        soil_file: Soil reflectance spectra file
        optipar_file: Fluspect optical parameters file
        atmo_file: Atmospheric data file
        directional_file: BRDF angles file (if calc_directional)
        timeseries_file: Time series input file (if simulation=1)
    """

    input_dir: str = "input"
    output_dir: str = "output"
    soil_file: str = "soil_spectra/soilnew.txt"
    optipar_file: str = "fluspect_parameters/Optipar2021_ProspectPRO.mat"
    atmo_file: str = "radiationdata/FLEX-S3_std.atm"
    directional_file: str = "directional/brdf_angles2.dat"
    timeseries_file: str = ""

    def get_full_path(self, file_key: str) -> str:
        """Get full path for a file relative to input directory.

        Args:
            file_key: File attribute name (e.g., 'soil_file')

        Returns:
            Full path to the file.
        """
        import os

        file_path = getattr(self, file_key)
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.input_dir, file_path)
