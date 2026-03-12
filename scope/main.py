"""Main entry point for SCOPE model.

This module provides the main simulation runner for SCOPE.
Translated from: SCOPE.m

The main workflow follows:
1. leafangles() - compute LIDF
2. fluspect() - compute leaf optical properties
3. BSM() - compute soil reflectance
4. RTMo() - optical radiative transfer
5. ebal() - energy balance iteration
6. RTMt_sb() - thermal radiation
7. RTMf() - fluorescence (optional)
8. RTMz() - xanthophyll/PRI (optional)
9. calc_brdf() - directional reflectance (optional)
10. Compute data products
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray

from .constants import CONSTANTS, TEMP_RESPONSE
from .spectral import SPECTRAL, SpectralBands
from .types import Angles, Canopy, LeafBio, Meteo, Options, Soil
from .supporting.leafangles import compute_canopy_lidf
from .supporting.meanleaf import meanleaf


@dataclass
class SCOPEOutput:
    """Output from a SCOPE simulation.

    Attributes:
        rad: Radiative transfer outputs
        gap: Gap probabilities
        thermal: Thermal outputs (temperatures)
        fluxes: Energy balance fluxes
        bcu: Biochemical output for sunlit leaves
        bch: Biochemical output for shaded leaves
        resistance: Aerodynamic resistances
        canopy: Updated canopy with computed properties
        iter_count: Number of energy balance iterations
        fluorescence: Fluorescence RTM output (if calc_fluor)
        xanthophyll: Xanthophyll/PRI RTM output (if calc_xanthophyll)
        directional: Directional (BRDF) output (if calc_directional)
    """
    rad: object = None
    gap: object = None
    thermal: object = None
    fluxes: dict = field(default_factory=dict)
    bcu: object = None
    bch: object = None
    resistance: object = None
    canopy: object = None
    iter_count: int = 0
    fluorescence: object = None
    xanthophyll: object = None
    directional: object = None


def load_matlab_atmosphere(
    spectral: SpectralBands,
    meteo: Meteo,
    data_path: str = None,
) -> Optional[dict]:
    """Load atmospheric data from MATLAB data files.

    Tries to load MODTRAN .atm file first (for full spectral features),
    falls back to simple Esun_.dat/Esky_.dat files.

    Args:
        spectral: Spectral band definitions
        meteo: Meteorological conditions with Rin and Rli
        data_path: Path to radiationdata folder (default: auto-detect)

    Returns:
        Dictionary with either:
        - 'M': MODTRAN matrix (6 columns) for full atmospheric RT
        - 'Esun_', 'Esky_': Simple spectral arrays
        Or None if files not found
    """
    import os
    from .supporting.integration import sint

    # Find data path
    if data_path is None:
        # Try to find relative to this module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(module_dir, '..', 'input', 'radiationdata'),
            os.path.join(module_dir, '..', '..', 'input', 'radiationdata'),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                data_path = p
                break

    if data_path is None:
        return None

    # Try MODTRAN .atm file first (has full atmospheric features)
    atm_file = os.path.join(data_path, 'FLEX-S3_std.atm')
    if os.path.exists(atm_file):
        try:
            from .io.load_atmo import load_atmo
            atmo = load_atmo(atm_file, spectral)
            return atmo  # Returns {'M': matrix}
        except Exception as e:
            print(f"Warning: Could not load MODTRAN file: {e}")

    # Fall back to simple Esun_/Esky_ files
    esun_file = os.path.join(data_path, 'Esun_.dat')
    esky_file = os.path.join(data_path, 'Esky_.dat')

    if not os.path.exists(esun_file) or not os.path.exists(esky_file):
        return None

    try:
        # Load MATLAB spectral data (these are reference spectra)
        Esun_ref = np.loadtxt(esun_file)
        Esky_ref = np.loadtxt(esky_file)

        # These files should have 2162 values matching wlS
        if len(Esun_ref) != len(spectral.wlS):
            return None

        # Calculate reference total incoming radiation
        # MATLAB uses 0.001 * sint(...) for integration
        wl = spectral.wlS
        J_o = wl < 3000  # Optical band

        ref_total = 0.001 * sint(Esun_ref[J_o] + Esky_ref[J_o], wl[J_o])

        if ref_total > 0 and meteo.Rin > 0:
            # Scale to match actual Rin
            scale = meteo.Rin / ref_total
            Esun_ = Esun_ref * scale
            Esky_ = Esky_ref * scale
        else:
            Esun_ = Esun_ref.copy()
            Esky_ = Esky_ref.copy()

        # Handle thermal region (Rli)
        J_t = wl >= 3000
        wl_t = wl[J_t]

        if len(wl_t) > 0 and meteo.Rli > 0:
            # Create thermal spectrum shape
            spectrum_t = np.exp(-0.5 * ((wl_t - 10000) / 5000)**2)
            spectrum_t_integral = sint(spectrum_t, wl_t)

            if spectrum_t_integral > 0:
                Esky_[J_t] = (meteo.Rli * 1000 / spectrum_t_integral) * spectrum_t
                Esun_[J_t] = 1e-10

        return {
            'Esun_': Esun_,
            'Esky_': Esky_,
        }

    except Exception:
        return None


def create_default_atmosphere(spectral: SpectralBands, meteo: Meteo) -> dict:
    """Create default atmospheric data when MODTRAN files not available.

    This function handles BOTH shortwave (Rin) and longwave (Rli) radiation
    spectrally, matching MATLAB's approach where thermal radiation is absorbed
    in RTMo (not RTMt).

    Note: MATLAB uses 0.001 * sint(...) for integration, so spectral values
    are stored in units that give correct total when multiplied by 0.001.

    Args:
        spectral: Spectral band definitions
        meteo: Meteorological conditions with Rin and Rli

    Returns:
        Dictionary with Esun_ and Esky_ spectral irradiance [mW/m²/nm]
    """
    # First try to load MATLAB spectral data files
    matlab_atmo = load_matlab_atmosphere(spectral, meteo)
    if matlab_atmo is not None:
        return matlab_atmo

    # Fall back to simplified approximation
    from .supporting.integration import sint

    wl = spectral.wlS
    nwl = len(wl)

    # Initialize output arrays with small values (not zero to avoid numerical issues)
    Esun_ = np.full(nwl, 1e-10)
    Esky_ = np.full(nwl, 1e-10)

    # === Optical/shortwave region (wl < 3000nm) ===
    J_o = wl < 3000  # Optical band indices

    # Create solar spectrum shape (rough approximation)
    # Peak around 500nm, declining toward UV and IR
    spectrum_o = np.exp(-0.5 * ((wl[J_o] - 500) / 300)**2)

    # Normalize so that 0.001 * sint(Esun_, wl) gives correct total
    spectrum_o_integral = sint(spectrum_o, wl[J_o])

    if spectrum_o_integral > 0 and meteo.Rin > 0:
        # Split into direct (70%) and diffuse (30%)
        Esun_[J_o] = (0.7 * meteo.Rin * 1000 / spectrum_o_integral) * spectrum_o
        Esky_[J_o] = (0.3 * meteo.Rin * 1000 / spectrum_o_integral) * spectrum_o

    # === Thermal/longwave region (wl >= 3000nm) ===
    # Match MATLAB: put Rli into Esky_ thermal bands (diffuse atmospheric thermal)
    J_t = wl >= 3000  # Thermal band indices
    wl_t = wl[J_t]

    if len(wl_t) > 0 and meteo.Rli > 0:
        # Create a thermal spectrum shape based on ~280K blackbody peak (~10um)
        # Peak around 10000nm (10um), declining toward shorter and longer wavelengths
        spectrum_t = np.exp(-0.5 * ((wl_t - 10000) / 5000)**2)

        # Normalize so that 0.001 * sint(Esky_thermal, wl_t) gives Rli
        spectrum_t_integral = sint(spectrum_t, wl_t)

        if spectrum_t_integral > 0:
            # All thermal radiation is diffuse (sky thermal)
            # Scale factor: Rli [W/m²] * 1000 [mW/W] / integral = spectral density
            Esky_[J_t] = (meteo.Rli * 1000 / spectrum_t_integral) * spectrum_t
            # No direct thermal from sun
            Esun_[J_t] = 1e-10

    return {
        'Esun_': Esun_,
        'Esky_': Esky_,
    }


def run_scope(
    leafbio: LeafBio,
    canopy: Canopy,
    soil: Soil,
    meteo: Meteo,
    angles: Angles,
    options: Optional[Options] = None,
    spectral: Optional[SpectralBands] = None,
    atmo: Optional[dict] = None,
) -> SCOPEOutput:
    """Run a complete SCOPE simulation.

    This is the main entry point for running SCOPE simulations.
    It orchestrates the radiative transfer, photosynthesis, and
    energy balance calculations following the MATLAB SCOPE.m workflow.

    Args:
        leafbio: Leaf biochemistry parameters
        canopy: Canopy structure parameters
        soil: Soil properties
        meteo: Meteorological conditions
        angles: Sun-observer geometry
        options: Simulation options (default: Options())
        spectral: Spectral band definitions (default: SPECTRAL)
        atmo: Atmospheric data (default: simple 70/30 direct/diffuse)

    Returns:
        SCOPEOutput containing all simulation results

    Example:
        >>> from scope import run_scope
        >>> from scope.types import LeafBio, Canopy, Soil, Meteo, Angles
        >>>
        >>> leafbio = LeafBio(Cab=40.0, Vcmax25=60.0)
        >>> canopy = Canopy(LAI=3.0, hc=2.0)
        >>> soil = Soil()
        >>> meteo = Meteo(Rin=600, Ta=25)
        >>> angles = Angles(tts=30)
        >>>
        >>> output = run_scope(leafbio, canopy, soil, meteo, angles)
        >>> print(f"GPP: {output.canopy.GPP:.2f} umol/m2/s")
    """
    # Import modules here to avoid circular imports
    from .rtm.fluspect import fluspect
    from .rtm.bsm import bsm
    from .rtm.rtmo import rtmo
    from .rtm.rtmt import rtmt_sb
    from .fluxes.ebal import ebal

    # Set defaults
    if options is None:
        options = Options()
    if spectral is None:
        spectral = SPECTRAL

    # Determine integration mode based on options
    integr = 'layers' if options.lite else 'angles'

    # 1. Setup canopy layers
    # MATLAB: canopy.nlayers = ceil(10*canopy.LAI) + ((meteo.Rin < 200) & options.MoninObukhov)*60
    nl_base = max(2, int(np.ceil(10 * canopy.LAI)))
    if meteo.Rin < 200:
        nl_base += 60  # More layers for low light (stability)
    canopy._nlayers = nl_base  # Set internal variable
    nl = canopy.nlayers
    # xl is computed automatically by Canopy class based on nlayers

    # 2. Compute leaf angle distribution
    compute_canopy_lidf(canopy)

    # 3. Leaf emissivity is computed automatically as property:
    # leafbio.emis = 1 - rho_thermal - tau_thermal

    # 4. Leaf optical properties using Fluspect
    try:
        from .rtm.fluspect import fluspect, LeafOpticalOutput
        from .io.load_optipar import load_optipar

        # Load optical parameters from data file
        optipar, wlP_data = load_optipar()

        # Run Fluspect model (include wlE and wlF for fluorescence if enabled)
        leafopt = fluspect(
            leafbio=leafbio,
            optipar=optipar,
            wlP=spectral.wlP,
            wlE=spectral.wlE if options.calc_fluor else None,
            wlF=spectral.wlF if options.calc_fluor else None,
        )

        # Interpolate from wlP to wlS if needed
        if len(leafopt.refl) == len(spectral.wlP):
            # Extend to full spectrum (wlS includes thermal)
            refl_full = np.interp(spectral.wlS, spectral.wlP, leafopt.refl, left=0.05, right=0.05)
            tran_full = np.interp(spectral.wlS, spectral.wlP, leafopt.tran, left=0.05, right=0.01)
            leafopt.refl = refl_full
            leafopt.tran = tran_full

    except FileNotFoundError:
        # Fall back to placeholder if data files not found
        from .rtm.fluspect import LeafOpticalOutput
        nwl = len(spectral.wlS)
        wl = spectral.wlS

        refl = np.where(wl < 700, 0.05,
               np.where(wl < 1300, 0.45,
               np.where(wl < 2500, 0.30, 0.05)))
        tran = np.where(wl < 700, 0.05,
               np.where(wl < 1300, 0.45,
               np.where(wl < 2500, 0.20, 0.01)))

        wlP = spectral.wlP
        kChlrel = np.where((wlP > 400) & (wlP < 700), 1.0, 0.0)
        kCarrel = np.where((wlP > 400) & (wlP < 550), 0.8, 0.0)

        leafopt = LeafOpticalOutput(
            refl=refl,
            tran=tran,
            kChlrel=kChlrel,
            kCarrel=kCarrel,
        )

    # Set thermal properties
    leafopt.refl[spectral.IwlT] = leafbio.rho_thermal
    leafopt.tran[spectral.IwlT] = leafbio.tau_thermal

    # Compute zeaxanthin spectra for xanthophyll/PRI calculations
    if options.calc_xanthophyll:
        try:
            from dataclasses import replace
            # Create leafbio copy with V2Z=1 (fully converted to zeaxanthin)
            leafbio_Z = replace(leafbio, V2Z=1.0)

            # Run Fluspect with full zeaxanthin conversion
            leafopt_Z = fluspect(
                leafbio=leafbio_Z,
                optipar=optipar,
                wlP=spectral.wlP,
            )

            # Interpolate to full spectrum (wlS)
            if len(leafopt_Z.refl) == len(spectral.wlP):
                reflZ_full = np.interp(spectral.wlS, spectral.wlP, leafopt_Z.refl, left=0.05, right=0.05)
                tranZ_full = np.interp(spectral.wlS, spectral.wlP, leafopt_Z.tran, left=0.05, right=0.01)
            else:
                reflZ_full = leafopt_Z.refl
                tranZ_full = leafopt_Z.tran

            # Store zeaxanthin spectra in leafopt
            leafopt.reflZ = reflZ_full
            leafopt.tranZ = tranZ_full
        except Exception:
            # If zeaxanthin calculation fails, RTMz will use unmodified values
            pass

    # 5. Soil reflectance using BSM model
    if soil.refl is None:
        try:
            from .rtm.bsm import bsm
            from .io.load_optipar import load_bsm_spectra

            # Load BSM spectral data
            bsm_spectra = load_bsm_spectra()

            # Run BSM model
            soil_refl_wlP = bsm(
                brightness=soil.BSMBrightness,
                lat=soil.BSMlat,
                lon=soil.BSMlon,
                SMC_fraction=soil.SMC,
                spectra=bsm_spectra,
            )

            # Interpolate to full spectrum (wlS)
            soil.refl = np.interp(spectral.wlS, spectral.wlP, soil_refl_wlP, left=0.1, right=0.1)

        except (FileNotFoundError, Exception):
            # Fall back to simple soil reflectance
            nwl = len(spectral.wlS)
            wl = spectral.wlS
            soil.refl = np.where(wl < 700, 0.10 + 0.0001 * wl,
                        np.where(wl < 2500, 0.25, 0.15))

    # Set thermal soil reflectance
    soil.refl[spectral.IwlT] = soil.rs_thermal

    # 6. Create atmospheric data if not provided
    if atmo is None:
        atmo = create_default_atmosphere(spectral, meteo)

    # 7. Optical radiative transfer (RTMo)
    # Convert Options dataclass to dict for rtmo
    from dataclasses import asdict
    options_dict = asdict(options)

    rad, gap, profiles = rtmo(
        spectral=spectral,
        atmo=atmo,
        soil=soil,
        leafopt=leafopt,
        canopy=canopy,
        angles=angles,
        meteo=meteo,
        options=options_dict,
    )

    # 8. Energy balance iteration (ebal)
    # This calls biochemical, resistances, heatfluxes, and RTMt internally
    iter_count, rad, thermal, soil, bcu, bch, fluxes, resistance, meteo = ebal(
        constants=CONSTANTS,
        options=options,
        rad=rad,
        gap=gap,
        meteo=meteo,
        soil=soil,
        canopy=canopy,
        leafbio=leafbio,
        k=1,
        xyt=None,
        integr=integr,
    )

    # 9. Final thermal radiation calculation
    thermal_out = rtmt_sb(
        rad=rad,
        soil=soil,
        leafbio=leafbio,
        canopy=canopy,
        gap=gap,
        Tcu=thermal['Tcu'],
        Tch=thermal['Tch'],
        Tsu=thermal['Tsu'],
        Tsh=thermal['Tsh'],
        constants=CONSTANTS,
        obsdir=True,
        spectral=spectral,
        Rli=meteo.Rli,
    )

    # Update rad with thermal outputs
    rad.Lot_ = thermal_out.Lot_
    rad.Eoutte_ = thermal_out.Eoutte_
    rad.Lote = thermal_out.Lote
    rad.LST = thermal_out.LST

    # 10. Fluorescence calculation (if enabled)
    fluor_output = None
    if options.calc_fluor and leafopt.Mb is not None:
        try:
            from .rtm.rtmf import rtmf

            # Get fluorescence efficiency from biochemical outputs
            etau = bcu.eta if hasattr(bcu, 'eta') else np.full(nl, 0.01)
            etah = bch.eta if hasattr(bch, 'eta') else np.full(nl, 0.01)

            fluor_output = rtmf(
                spectral=spectral,
                rad=rad,
                soil=soil,
                leafopt=leafopt,
                canopy=canopy,
                gap=gap,
                angles=angles,
                etau=etau,
                etah=etah,
            )

            # Store fluorescence in rad
            rad.LoF_ = fluor_output.LoF_
            rad.EoutF_ = fluor_output.EoutF_
            rad.EoutF = fluor_output.EoutF
            rad.LoutF = fluor_output.LoutF
            rad.F685 = fluor_output.F685
            rad.F740 = fluor_output.F740

        except (ImportError, Exception) as e:
            # Fluorescence calculation failed, continue without it
            pass

    # 11. Xanthophyll/PRI calculation (if enabled)
    xanth_output = None
    if options.calc_xanthophyll and hasattr(leafopt, 'reflZ') and leafopt.reflZ is not None:
        try:
            from .rtm.rtmz import rtmz

            # Get NPQ rate constant from biochemical outputs
            Knu = bcu.Kn if hasattr(bcu, 'Kn') else np.full(nl, 0.0)
            Knh = bch.Kn if hasattr(bch, 'Kn') else np.full(nl, 0.0)

            xanth_output = rtmz(
                spectral=spectral,
                rad=rad,
                soil=soil,
                leafopt=leafopt,
                canopy=canopy,
                gap=gap,
                angles=angles,
                Knu=Knu,
                Knh=Knh,
            )

            # Update rad with xanthophyll-modified values
            rad.Lo_ = xanth_output.Lo_mod
            rad.refl = xanth_output.refl_mod
            rad.rso = xanth_output.rso_mod
            rad.rdo = xanth_output.rdo_mod
            rad.Eout_ = xanth_output.Eout_mod
            rad.PRI = xanth_output.PRI

        except (ImportError, Exception):
            # Xanthophyll calculation failed, continue without it
            pass

    # 12. Directional (BRDF) calculation (if enabled)
    dir_output = None
    if options.calc_directional:
        try:
            from .supporting.brdf import calc_brdf

            dir_output = calc_brdf(
                spectral=spectral,
                atmo=atmo,
                soil=soil,
                leafopt=leafopt,
                canopy=canopy,
                angles=angles,
                meteo=meteo,
                options=options_dict,
                thermal=thermal,
                bcu=bcu,
                bch=bch,
            )

        except (ImportError, Exception):
            # Directional calculation failed, continue without it
            pass

    # 13. Compute data products
    Ps = gap.Ps[:nl]
    Ph = 1 - Ps
    LAI = canopy.LAI
    lidf = canopy.lidf
    nlazi = canopy.nlazi

    # Sunlit/shaded LAI
    canopy.LAIsunlit = LAI * np.mean(Ps)
    canopy.LAIshaded = LAI - canopy.LAIsunlit

    # Determine correct meanleaf choice for canopy-integrated values (scalars)
    # For 3D sunlit arrays, use 'angles_and_layers' to get scalar output
    sunlit_choice = 'angles_and_layers' if (integr == 'angles' and rad.Pnu.ndim == 3) else 'layers'

    # Absorbed PAR by chlorophyll
    canopy.Pnsun_Cab = LAI * meanleaf(rad.Pnu_Cab, lidf, nlazi, sunlit_choice, Ps)
    canopy.Pnsha_Cab = LAI * meanleaf(rad.Pnh_Cab, lidf, nlazi, 'layers', Ph)
    canopy.Pntot_Cab = canopy.Pnsun_Cab + canopy.Pnsha_Cab

    # Absorbed PAR total
    canopy.Pnsun = LAI * meanleaf(rad.Pnu, lidf, nlazi, sunlit_choice, Ps)
    canopy.Pnsha = LAI * meanleaf(rad.Pnh, lidf, nlazi, 'layers', Ph)
    canopy.Pntot = canopy.Pnsun + canopy.Pnsha

    # Land surface temperature
    canopy.LST = thermal_out.LST
    canopy.emis = rad.canopyemis if hasattr(rad, 'canopyemis') else 0.98

    # Photosynthesis (from biochemical outputs)
    # Use same sunlit_choice for 3D arrays to get scalar output
    bcu_choice = 'angles_and_layers' if (integr == 'angles' and bcu.A.ndim == 3) else 'layers'
    canopy.A = LAI * (
        meanleaf(bch.A, lidf, nlazi, 'layers', Ph) +
        meanleaf(bcu.A, lidf, nlazi, bcu_choice, Ps)
    )
    canopy.GPP = LAI * (
        meanleaf(bch.Ag, lidf, nlazi, 'layers', Ph) +
        meanleaf(bcu.Ag, lidf, nlazi, bcu_choice, Ps)
    )

    # Electron transport
    canopy.Ja = LAI * (
        meanleaf(bch.Ja, lidf, nlazi, 'layers', Ph) +
        meanleaf(bcu.Ja, lidf, nlazi, bcu_choice, Ps)
    )

    # Package output
    output = SCOPEOutput(
        rad=rad,
        gap=gap,
        thermal=thermal,
        fluxes=fluxes,
        bcu=bcu,
        bch=bch,
        resistance=resistance,
        canopy=canopy,
        iter_count=iter_count,
        fluorescence=fluor_output,
        xanthophyll=xanth_output,
        directional=dir_output,
    )

    return output


def main():
    """Command-line entry point for SCOPE."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SCOPE: Soil Canopy Observation, Photochemistry and Energy fluxes"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="SCOPE Python 0.1.0",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test simulation with default parameters",
    )

    args = parser.parse_args()

    if args.test:
        print("Running test simulation...")

        # Create default inputs
        leafbio = LeafBio(Cab=40.0, Vcmax25=60.0)
        canopy = Canopy(LAI=3.0, hc=2.0)
        soil = Soil()
        meteo = Meteo(Rin=600, Ta=25)
        angles = Angles(tts=30)

        # Run simulation
        try:
            output = run_scope(leafbio, canopy, soil, meteo, angles)

            print("Test simulation completed successfully!")
            print(f"  LAI: {canopy.LAI}")
            print(f"  Cab: {leafbio.Cab} µg/cm²")
            print(f"  Solar zenith: {angles.tts}°")
            print(f"  Iterations: {output.iter_count}")
            if hasattr(output.canopy, 'GPP'):
                print(f"  GPP: {output.canopy.GPP:.2f} µmol/m²/s")
            if hasattr(output.canopy, 'A'):
                print(f"  Net photosynthesis: {output.canopy.A:.2f} µmol/m²/s")
        except Exception as e:
            print(f"Test simulation failed: {e}")
            import traceback
            traceback.print_exc()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
