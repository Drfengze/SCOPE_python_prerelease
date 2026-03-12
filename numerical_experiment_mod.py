#!/usr/bin/env python3
"""
Numerical Experiment - MODTRAN Version.

This version uses MODTRAN atmospheric data for realistic spectral irradiance,
instead of the simplified 70/30 direct/diffuse Gaussian model.

Based on numerical_experiment_simple.py and scope_main.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
from itertools import product
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '.')

from scope.spectral import SpectralBands
from scope.constants import CONSTANTS
from scope.types import LeafBio, Canopy, Soil, Meteo, Angles, Options
from scope.io.load_optipar import load_optipar
from scope.io.load_atmo import load_atmo, aggreg
from scope.rtm.fluspect import fluspect
from scope.rtm.rtmo import rtmo
from scope.rtm.rtmf import rtmf
from scope.fluxes.ebal import ebal
from scope.supporting.leafangles import compute_canopy_lidf, leafangles


# Parameter values (Table 2) - matches MATLAB numerical_experiment_matlab_mod.m
Cab_values = [10, 20, 40, 80]
LAI_values = [0.5, 1, 3, 6]
LAD_types = ['spherical', 'planophile', 'erectophile']
tts_values = [30, 45, 60]
tto_values = [0, 20, 40, 60]
Rin_values = [100, 300, 500, 800]
Ta_values = [15, 25, 35]
Vcmax_values = [30, 100, 160]
soil_types = ['zero', 'wet', 'dry_bright1', 'dry_bright2']

# LIDF parameters for each LAD type
LIDF_params = {
    'spherical': (-0.35, -0.15),
    'planophile': (1.0, 0.0),
    'erectophile': (-1.0, 0.0),
}

# Soil reflectance values
soil_refl_values = {
    'zero': 0.0,
    'wet': 0.05,
    'dry_bright1': 0.15,
    'dry_bright2': 0.25,
}


# --- Worker process globals (set once per worker via initializer) ---
_worker_spectral = None
_worker_optipar = None
_worker_atmo = None


def _init_worker(spectral, optipar, atmo):
    global _worker_spectral, _worker_optipar, _worker_atmo
    _worker_spectral = spectral
    _worker_optipar = optipar
    _worker_atmo = atmo


def _run_scenario(params):
    """Worker function: run a single scenario and return results dict.

    Args:
        params: tuple of (Cab, LAI, LAD, tts, tto, Rin, Ta, Vcmax, soil_type, soil_refl_val, group)

    Returns:
        dict with scalar results and spectral arrays
    """
    Cab, LAI, LAD, tts, tto, Rin, Ta, Vcmax, soil_type, soil_refl_val, group = params
    spectral = _worker_spectral
    optipar = _worker_optipar
    atmo = _worker_atmo

    base = {
        'Cab': Cab, 'LAI': LAI, 'LAD': LAD, 'tts': tts,
        'tto': tto, 'Rin': Rin, 'Ta': Ta, 'Vcmax': Vcmax,
        'soil_type': soil_type, 'group': group,
    }
    nan_result = {
        **base,
        'F684': np.nan, 'F685': np.nan, 'F740': np.nan, 'F761': np.nan,
        'wl685': np.nan, 'wl740': np.nan,
        'LoutF': np.nan, 'EoutF': np.nan, 'Eouto': np.nan,
        'escape_685': np.nan, 'escape_740': np.nan, 'escape_761': np.nan,
        'Rntot': np.nan, 'lEtot': np.nan, 'Htot': np.nan,
        'Actot': np.nan, 'Rnctot': np.nan, 'Tcave': np.nan,
        'Tsave': np.nan, 'error': '',
        'spec_refl': None, 'spec_Lo_': None, 'spec_Eout_': None,
        'spec_Esun_': None, 'spec_Esky_': None, 'spec_LoF_': None,
        'spec_EoutF_': None, 'spec_LoF_sunlit': None,
        'spec_LoF_shaded': None, 'spec_LoF_scattered': None,
        'spec_LoF_soil': None, 'spec_Femleaves_': None,
        'spec_escape_ratio_': None,
    }

    try:
        lidf = LIDF_params[LAD]

        leafbio = LeafBio(
            Cab=Cab, Cca=Cab / 4, Cdm=0.012, Cw=0.009,
            Cs=0.0, Cant=0.0, Cp=0.0, Cbc=0.0, N=1.4,
            fqe=0.01, V2Z=0, Vcmax25=Vcmax,
            BallBerrySlope=9.0, BallBerry0=0.01, Type='C3',
            RdPerVcmax25=0.015, Kn0=2.48, Knalpha=2.83, Knbeta=0.114,
            Tyear=15, beta=0.51, kNPQs=0, qLs=0, stressfactor=1,
            rho_thermal=0.01, tau_thermal=0.01,
        )

        hc = 2.0
        leafwidth = 0.1
        nlayers = max(2, int(np.ceil(10 * LAI)))
        canopy = Canopy(
            LAI=LAI, hc=hc, LIDFa=lidf[0], LIDFb=lidf[1],
            leafwidth=leafwidth, kV=0.6396, rb=10, Cd=0.3, CR=0.35,
            CD1=20.6, Psicor=0.2, rwc=1, zo=0.25 * hc, d=0.65 * hc,
        )
        canopy._nlayers = nlayers
        canopy._x = -np.arange(1, nlayers + 1) / nlayers
        canopy._xl = np.concatenate([[0], canopy._x])

        nwl = len(spectral.wlS)
        soil = Soil(
            spectrum=1, rs_thermal=0.06, SMC=0.25, rss=500.0, rbs=10.0,
            cs=1180.0, rhos=1800.0, CSSOIL=0.01, lambdas=1.55,
            BSMBrightness=0.5, BSMlat=25.0, BSMlon=45.0,
        )
        soil.Tsold = Ta * np.ones((12, 2))
        soil_refl = soil_refl_val * np.ones(nwl)
        soil_refl[spectral.IwlT] = soil.rs_thermal
        soil.refl = soil_refl

        meteo = Meteo(z=10.0, Rin=Rin, Rli=300.0, Ta=Ta, p=970.0,
                      ea=15.0, u=2.0, Ca=410.0, Oa=209.0)

        angles = Angles(tts=tts, tto=tto, psi=0.0)

        options = Options(
            calc_fluor=True, calc_planck=False, calc_xanthophyll=False,
            Fluorescence_model=0, calc_directional=False,
            calc_vert_profiles=False, calc_ebal=True, lite=True,
            verify=False, apply_T_corr=True, MoninObukhov=False,
            soil_heat_method=2, calc_rss_rbs=False,
        )

        output = run_single_scenario(
            leafbio=leafbio, canopy=canopy, soil=soil, meteo=meteo,
            angles=angles, options=options, spectral=spectral,
            optipar=optipar, atmo=atmo,
        )

        # Extract fluorescence
        if output['fluorescence'] is not None:
            fl = output['fluorescence']
            F684 = getattr(fl, 'F684', np.nan)
            F685 = fl.F685
            F740 = fl.F740
            F761 = getattr(fl, 'F761', np.nan)
            wl685 = getattr(fl, 'wl685', np.nan)
            wl740 = getattr(fl, 'wl740', np.nan)
            LoutF = fl.LoutF
            EoutF = fl.EoutF
            # Escape ratio = EoutF_ / Femleaves_ (per wavelength)
            with np.errstate(divide='ignore', invalid='ignore'):
                escape_ratio_ = np.where(fl.Femleaves_ > 0,
                                         fl.EoutF_ / fl.Femleaves_, np.nan)
            # Escape ratio at 685, 740, 761 nm (wlF starts at 640)
            escape_685 = escape_ratio_[45] if len(escape_ratio_) > 45 else np.nan
            escape_740 = escape_ratio_[100] if len(escape_ratio_) > 100 else np.nan
            escape_761 = escape_ratio_[121] if len(escape_ratio_) > 121 else np.nan
        else:
            F684 = F685 = F740 = F761 = wl685 = wl740 = LoutF = EoutF = np.nan
            escape_685 = escape_740 = escape_761 = np.nan
            escape_ratio_ = None

        rad = output['rad']
        fluxes = output['fluxes']
        Eouto = getattr(rad, 'Eouto', np.nan) if rad else np.nan

        spec = output['spectral']

        result = {
            **base,
            'F684': F684, 'F685': F685, 'F740': F740, 'F761': F761,
            'wl685': wl685, 'wl740': wl740,
            'LoutF': LoutF, 'EoutF': EoutF, 'Eouto': Eouto,
            'escape_685': escape_685, 'escape_740': escape_740, 'escape_761': escape_761,
            'Rntot': fluxes.get('Rntot', np.nan),
            'lEtot': fluxes.get('lEtot', np.nan),
            'Htot': fluxes.get('Htot', np.nan),
            'Actot': fluxes.get('Actot', np.nan),
            'Rnctot': fluxes.get('Rnctot', np.nan),
            'Tcave': fluxes.get('Tcave', np.nan),
            'Tsave': fluxes.get('Tsave', np.nan),
            'error': '',
            'spec_refl': spec['refl'],
            'spec_Lo_': spec['Lo_'],
            'spec_Eout_': spec['Eout_'],
            'spec_Esun_': spec['Esun_'],
            'spec_Esky_': spec['Esky_'],
            'spec_LoF_': spec['LoF_'],
            'spec_EoutF_': spec['EoutF_'],
            'spec_LoF_sunlit': spec['LoF_sunlit'],
            'spec_LoF_shaded': spec['LoF_shaded'],
            'spec_LoF_scattered': spec['LoF_scattered'],
            'spec_LoF_soil': spec['LoF_soil'],
            'spec_Femleaves_': spec['Femleaves_'],
            'spec_escape_ratio_': escape_ratio_,
        }
        return result

    except Exception as e:
        nan_result['error'] = str(e)
        return nan_result


def load_modtran_atmo(spectral, atmfile=None):
    """Load MODTRAN atmospheric data.

    Args:
        spectral: SpectralBands instance
        atmfile: Path to .atm file. If None, uses default FLEX-S3_std.atm

    Returns:
        Dictionary with 'M' key containing aggregated MODTRAN matrix
    """
    if atmfile is None:
        # Default MODTRAN file
        atmfile = str(Path(__file__).parent / 'input' / 'radiationdata' / 'FLEX-S3_std.atm')

    atmfile = Path(atmfile)
    if not atmfile.exists():
        raise FileNotFoundError(f"MODTRAN file not found: {atmfile}")

    # Load and aggregate MODTRAN data
    atmo = load_atmo(str(atmfile), spectral)
    return atmo


def run_single_scenario(leafbio, canopy, soil, meteo, angles, options,
                        spectral, optipar, atmo):
    """Run SCOPE for a single scenario with pre-loaded data.

    This version uses MODTRAN atmospheric data for spectral irradiance.
    The rtmo function handles the conversion from MODTRAN matrix to
    Esun_/Esky_ arrays, scaling by meteo.Rin and meteo.Rli.
    """
    from dataclasses import asdict

    # Setup canopy layers
    nl = canopy.nlayers
    integr = 'layers' if options.lite else 'angles'

    # Compute leaf angle distribution
    compute_canopy_lidf(canopy)

    # Run Fluspect (leaf optical properties)
    leafopt = fluspect(
        leafbio=leafbio,
        optipar=optipar,
        wlP=spectral.wlP,
        wlE=spectral.wlE if options.calc_fluor else None,
        wlF=spectral.wlF if options.calc_fluor else None,
    )

    # Extend to full spectrum if needed
    if len(leafopt.refl) == len(spectral.wlP):
        refl_full = np.interp(spectral.wlS, spectral.wlP, leafopt.refl, left=0.05, right=0.05)
        tran_full = np.interp(spectral.wlS, spectral.wlP, leafopt.tran, left=0.05, right=0.01)
        leafopt.refl = refl_full
        leafopt.tran = tran_full

    # Set thermal properties
    leafopt.refl[spectral.IwlT] = leafbio.rho_thermal
    leafopt.tran[spectral.IwlT] = leafbio.tau_thermal

    # RTMo (optical radiative transfer)
    # When atmo contains 'M' (MODTRAN data), rtmo calculates Esun_ and Esky_
    # from the atmospheric transmission data and scales to meteo.Rin
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

    # Energy balance
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

    # RTMf (fluorescence)
    fluor_output = None
    if options.calc_fluor and leafopt.Mb is not None:
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

    # Extract spectral outputs
    spectral_out = {
        'refl': rad.refl if hasattr(rad, 'refl') else None,
        'Lo_': rad.Lo_ if hasattr(rad, 'Lo_') else None,
        'Eout_': rad.Eout_ if hasattr(rad, 'Eout_') else None,
        'Esun_': rad.Esun_ if hasattr(rad, 'Esun_') else None,
        'Esky_': rad.Esky_ if hasattr(rad, 'Esky_') else None,
        'LoF_': fluor_output.LoF_ if fluor_output is not None and hasattr(fluor_output, 'LoF_') else None,
        'EoutF_': fluor_output.EoutF_ if fluor_output is not None and hasattr(fluor_output, 'EoutF_') else None,
        'LoF_sunlit': fluor_output.LoF_sunlit if fluor_output is not None and hasattr(fluor_output, 'LoF_sunlit') else None,
        'LoF_shaded': fluor_output.LoF_shaded if fluor_output is not None and hasattr(fluor_output, 'LoF_shaded') else None,
        'LoF_scattered': fluor_output.LoF_scattered if fluor_output is not None and hasattr(fluor_output, 'LoF_scattered') else None,
        'LoF_soil': fluor_output.LoF_soil if fluor_output is not None and hasattr(fluor_output, 'LoF_soil') else None,
        'Femleaves_': fluor_output.Femleaves_ if fluor_output is not None and hasattr(fluor_output, 'Femleaves_') else None,
    }

    return {
        'rad': rad,
        'thermal': thermal,
        'fluxes': fluxes,
        'fluorescence': fluor_output,
        'spectral': spectral_out,
    }


def run_experiment(output_dir=None, max_scenarios=None, atmfile=None, nworkers=None):
    """Run the full numerical experiment using MODTRAN atmosphere.

    Args:
        output_dir: Directory to save results
        max_scenarios: Maximum number of scenarios to run (for testing)
        atmfile: Path to MODTRAN .atm file. If None, uses default.
        nworkers: Number of parallel workers. Default: mp.cpu_count().
    """

    if output_dir is None:
        output_dir = 'output'
    if nworkers is None:
        nworkers = mp.cpu_count()

    os.makedirs(output_dir, exist_ok=True)

    # Build parameter grid
    param_grid = []
    for soil_idx, soil_type in enumerate(soil_types):
        soil_refl_val = soil_refl_values[soil_type]
        group = 1 if soil_idx == 0 else 2
        for combo in product(Cab_values, LAI_values, LAD_types, tts_values,
                             tto_values, Rin_values, Ta_values, Vcmax_values):
            Cab, LAI, LAD, tts, tto, Rin, Ta, Vcmax = combo
            param_grid.append((Cab, LAI, LAD, tts, tto, Rin, Ta, Vcmax,
                               soil_type, soil_refl_val, group))

    n_total = len(param_grid)
    if max_scenarios is not None:
        param_grid = param_grid[:max_scenarios]
    n_run = len(param_grid)
    print(f'Total scenarios: {n_total}, running: {n_run}, workers: {nworkers}')

    # Initialize spectral - LOAD ONCE
    spectral = SpectralBands()
    print('Loaded spectral bands')

    # Load optipar ONCE outside the loop
    print('Loading optipar...')
    optipar, wlP = load_optipar()
    print('Loaded optipar')

    # Load MODTRAN atmosphere ONCE outside the loop
    print('Loading MODTRAN atmospheric data...')
    atmo = load_modtran_atmo(spectral, atmfile)
    print(f'Loaded MODTRAN data with M matrix shape: {atmo["M"].shape}')

    # Preallocate results
    results = []

    # Preallocate spectral data arrays (sized to n_run, not n_total)
    nwl = len(spectral.wlS)
    nwlF = len(spectral.wlF)
    spectral_data = {
        'wl': spectral.wlS,
        'wlF': spectral.wlF,
        'refl': np.full((n_run, nwl), np.nan),
        'Lo_': np.full((n_run, nwl), np.nan),
        'Eout_': np.full((n_run, nwl), np.nan),
        'Esun_': np.full((n_run, nwl), np.nan),
        'Esky_': np.full((n_run, nwl), np.nan),
        'LoF_': np.full((n_run, nwlF), np.nan),
        'EoutF_': np.full((n_run, nwlF), np.nan),
        'LoF_sunlit': np.full((n_run, nwlF), np.nan),
        'LoF_shaded': np.full((n_run, nwlF), np.nan),
        'LoF_scattered': np.full((n_run, nwlF), np.nan),
        'LoF_soil': np.full((n_run, nwlF), np.nan),
        'Femleaves_': np.full((n_run, nwlF), np.nan),
        'escape_ratio_': np.full((n_run, nwlF), np.nan),
    }
    print(f'Preallocated spectral arrays: {n_run} scenarios x {nwl} wavelengths')

    # Run experiment in parallel
    start_time = time.time()
    errors = 0
    done = 0

    pool = mp.Pool(nworkers, initializer=_init_worker,
                   initargs=(spectral, optipar, atmo))
    try:
        for result in pool.imap(_run_scenario, param_grid):
            idx = done  # row index for spectral arrays
            done += 1

            if result['error']:
                errors += 1

            # Store spectral arrays and strip them from scalar dict
            for key, spec_key in [('refl', 'spec_refl'), ('Lo_', 'spec_Lo_'),
                                   ('Eout_', 'spec_Eout_'), ('Esun_', 'spec_Esun_'),
                                   ('Esky_', 'spec_Esky_'), ('LoF_', 'spec_LoF_'),
                                   ('EoutF_', 'spec_EoutF_'), ('LoF_sunlit', 'spec_LoF_sunlit'),
                                   ('LoF_shaded', 'spec_LoF_shaded'), ('LoF_scattered', 'spec_LoF_scattered'),
                                   ('LoF_soil', 'spec_LoF_soil'), ('Femleaves_', 'spec_Femleaves_'),
                                   ('escape_ratio_', 'spec_escape_ratio_')]:
                arr = result.pop(spec_key)
                if arr is not None:
                    spectral_data[key][idx, :] = arr

            results.append(result)

            # Progress
            if done % 500 == 0 or done == n_run:
                elapsed = time.time() - start_time
                rate = done / elapsed
                eta = (n_run - done) / rate / 60 if rate > 0 else 0
                print(f'  {done}/{n_run} ({100*done/n_run:.1f}%) - {rate:.1f}/s - ETA: {eta:.1f} min')
    finally:
        pool.close()
        pool.join()

    # Save results
    elapsed = time.time() - start_time
    print(f'\n=== DONE ===')
    print(f'Total: {done}, Errors: {errors}, Time: {elapsed/60:.1f} min')

    # Save scalar results to CSV
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(output_dir, f'numerical_experiment_modtran_{timestamp}.csv')
    df.to_csv(outfile, index=False)
    print(f'Saved scalar results: {outfile}')

    # Save spectral data to NPZ file
    specfile = os.path.join(output_dir, f'numerical_experiment_modtran_spectral_{timestamp}.npz')
    np.savez_compressed(
        specfile,
        wl=spectral_data['wl'],
        wlF=spectral_data['wlF'],
        refl=spectral_data['refl'],
        Lo_=spectral_data['Lo_'],
        Eout_=spectral_data['Eout_'],
        Esun_=spectral_data['Esun_'],
        Esky_=spectral_data['Esky_'],
        LoF_=spectral_data['LoF_'],
        EoutF_=spectral_data['EoutF_'],
        LoF_sunlit=spectral_data['LoF_sunlit'],
        LoF_shaded=spectral_data['LoF_shaded'],
        LoF_scattered=spectral_data['LoF_scattered'],
        LoF_soil=spectral_data['LoF_soil'],
        Femleaves_=spectral_data['Femleaves_'],
        escape_ratio_=spectral_data['escape_ratio_'],
    )
    print(f'Saved spectral data: {specfile}')
    print(f'  Spectral arrays size: {done} scenarios x {len(spectral_data["wl"])} wavelengths')
    print(f'  Fluorescence arrays size: {done} scenarios x {len(spectral_data["wlF"])} wavelengths')

    return df, spectral_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run SCOPE numerical experiment with MODTRAN atmosphere')
    parser.add_argument('--max-scenarios', type=int, default=None,
                        help='Maximum scenarios to run (for testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--atmfile', type=str, default=None,
                        help='Path to MODTRAN .atm file (default: FLEX-S3_std.atm)')
    parser.add_argument('--nworkers', type=int, default=None,
                        help='Number of parallel workers (default: all CPU cores)')

    args = parser.parse_args()

    df, spectral_data = run_experiment(
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios,
        atmfile=args.atmfile,
        nworkers=args.nworkers,
    )
