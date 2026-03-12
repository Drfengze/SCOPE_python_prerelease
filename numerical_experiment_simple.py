#!/usr/bin/env python3
"""
Numerical Experiment - Simplified Version using SCOPE's built-in structure.

This version uses SCOPE's existing infrastructure more directly.
Based on numerical_experiment_matlab_simple.m

Optimized: loads optipar and spectral data once outside the loop.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '.')

from scope.spectral import SpectralBands
from scope._paths import get_default_input_dir
from scope.constants import CONSTANTS
from scope.types import LeafBio, Canopy, Soil, Meteo, Angles, Options
from scope.io.load_optipar import load_optipar
from scope.rtm.fluspect import fluspect
from scope.rtm.rtmo import rtmo
from scope.rtm.rtmf import rtmf
from scope.fluxes.ebal import ebal
from scope.supporting.leafangles import compute_canopy_lidf, leafangles


# Parameter values (Table 2)
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


def create_simple_atmo(spectral, Rin, Rli):
    """Create simple atmospheric irradiance (70/30 direct/diffuse)."""
    nwl = len(spectral.wlS)
    Esun_ = np.zeros(nwl)
    Esky_ = np.zeros(nwl)

    wl = spectral.wlS

    # Optical region (wl < 3000 nm)
    iwl = wl < 3000
    if np.any(iwl) and Rin > 0:
        wl_opt = wl[iwl]
        spec = np.exp(-0.5 * ((wl_opt - 500) / 300) ** 2)
        tot = np.trapz(spec, wl_opt)
        if tot > 0:
            Esun_[iwl] = 0.7 * Rin * 1000 / tot * spec
            Esky_[iwl] = 0.3 * Rin * 1000 / tot * spec

    # Thermal region (wl >= 3000 nm)
    iwl = wl >= 3000
    if np.any(iwl) and Rli > 0:
        wl_therm = wl[iwl]
        spec = np.exp(-0.5 * ((wl_therm - 10000) / 5000) ** 2)
        tot = np.trapz(spec, wl_therm)
        if tot > 0:
            Esky_[iwl] = Rli * 1000 / tot * spec
            Esun_[iwl] = 1e-10

    return {
        'Esun_': Esun_,
        'Esky_': Esky_,
    }


def run_single_scenario(leafbio, canopy, soil, meteo, angles, options,
                        spectral, optipar, atmo):
    """Run SCOPE for a single scenario with pre-loaded data.

    This avoids reloading optipar on every call.
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

    return {
        'rad': rad,
        'thermal': thermal,
        'fluxes': fluxes,
        'fluorescence': fluor_output,
    }


def run_experiment(output_dir=None, max_scenarios=None, input_dir=None):
    """Run the full numerical experiment."""

    if output_dir is None:
        output_dir = 'output'

    os.makedirs(output_dir, exist_ok=True)

    # Calculate total
    n_total = (len(Cab_values) * len(LAI_values) * len(LAD_types) *
               len(tts_values) * len(tto_values) * len(Rin_values) *
               len(Ta_values) * len(Vcmax_values) * len(soil_types))
    print(f'Total scenarios: {n_total}')

    # Initialize spectral - LOAD ONCE
    spectral = SpectralBands()
    print('Loaded spectral bands')

    # Load optipar ONCE outside the loop
    print('Loading optipar...')
    if input_dir is None:
        input_dir = get_default_input_dir()
    optipar, wlP = load_optipar(input_dir=input_dir)
    print('Loaded optipar')

    # Preallocate results
    results = []

    # Run experiment
    idx = 0
    start_time = time.time()
    errors = 0

    for soil_idx, soil_type in enumerate(soil_types):
        soil_refl_val = soil_refl_values[soil_type]
        group = 1 if soil_idx == 0 else 2

        print(f'\n=== Soil: {soil_type} (group {group}) ===')

        for Cab in Cab_values:
            for LAI in LAI_values:
                for LAD in LAD_types:
                    lidf = LIDF_params[LAD]
                    for tts in tts_values:
                        for tto in tto_values:
                            for Rin in Rin_values:
                                for Ta in Ta_values:
                                    for Vcmax in Vcmax_values:
                                        idx += 1

                                        if max_scenarios is not None and idx > max_scenarios:
                                            break

                                        try:
                                            # Setup input structures using dataclasses
                                            # Match MATLAB numerical_experiment_matlab_simple.m settings
                                            leafbio = LeafBio(
                                                Cab=Cab,
                                                Cca=Cab / 4,
                                                Cdm=0.012,
                                                Cw=0.009,
                                                Cs=0.0,
                                                Cant=0.0,
                                                Cp=0.0,
                                                Cbc=0.0,
                                                N=1.4,
                                                fqe=0.01,
                                                V2Z=0,
                                                Vcmax25=Vcmax,
                                                BallBerrySlope=9.0,
                                                BallBerry0=0.01,
                                                Type='C3',
                                                RdPerVcmax25=0.015,
                                                Kn0=2.48,
                                                Knalpha=2.83,
                                                Knbeta=0.114,
                                                Tyear=15,
                                                beta=0.51,
                                                kNPQs=0,
                                                qLs=0,
                                                stressfactor=1,
                                                rho_thermal=0.01,
                                                tau_thermal=0.01,
                                            )

                                            # Match MATLAB canopy settings
                                            hc = 2.0
                                            leafwidth = 0.1
                                            nlayers = max(2, int(np.ceil(10 * LAI)))  # MATLAB: max(2, ceil(10*LAI))
                                            canopy = Canopy(
                                                LAI=LAI,
                                                hc=hc,
                                                LIDFa=lidf[0],
                                                LIDFb=lidf[1],
                                                leafwidth=leafwidth,
                                                rb=10,
                                                Cd=0.3,
                                                CR=0.35,
                                                CD1=20.6,
                                                Psicor=0.2,
                                                rwc=1,  # MATLAB: 1, Python default: 0.0
                                                zo=0.25 * hc,  # MATLAB: 0.25*hc, Python default: 0.1*hc
                                                d=0.65 * hc,   # MATLAB: 0.65*hc, Python default: 0.7*hc
                                            )
                                            # Override nlayers to match MATLAB formula
                                            canopy._nlayers = nlayers
                                            # Recompute layer arrays
                                            canopy._x = -np.arange(1, nlayers + 1) / nlayers
                                            canopy._xl = np.concatenate([[0], canopy._x])

                                            # Create soil and set reflectance - match MATLAB settings
                                            soil = Soil(
                                                spectrum=1,
                                                rs_thermal=0.06,
                                                SMC=0.25,
                                                rss=500.0,
                                                rbs=10.0,
                                                cs=1180.0,
                                                rhos=1800.0,
                                                CSSOIL=0.01,
                                                lambdas=1.55,
                                                BSMBrightness=0.5,
                                                BSMlat=25.0,
                                                BSMlon=45.0,
                                            )
                                            # Set Tsold to Ta * ones(12,2) as in MATLAB
                                            soil.Tsold = Ta * np.ones((12, 2))
                                            # Set soil reflectance after creation (it's a property)
                                            nwl = len(spectral.wlS)
                                            soil_refl = soil_refl_val * np.ones(nwl)
                                            soil_refl[spectral.IwlT] = soil.rs_thermal  # thermal
                                            soil.refl = soil_refl

                                            meteo = Meteo(
                                                z=10.0,       # measurement height [m]
                                                Rin=Rin,
                                                Rli=300.0,
                                                Ta=Ta,
                                                p=970.0,
                                                ea=15.0,
                                                u=2.0,
                                                Ca=410.0,
                                                Oa=209.0,     # O2 concentration [per mille]
                                            )

                                            angles = Angles(
                                                tts=tts,
                                                tto=tto,
                                                psi=0.0,
                                            )

                                            # Match MATLAB options settings
                                            options = Options(
                                                calc_fluor=True,
                                                calc_planck=False,        # MATLAB: 0 (use Stefan-Boltzmann)
                                                calc_xanthophyll=False,   # MATLAB: 0
                                                Fluorescence_model=0,     # MATLAB: 0 (empirical)
                                                calc_directional=False,   # MATLAB: 0
                                                calc_vert_profiles=False, # MATLAB: 0
                                                calc_ebal=True,           # MATLAB: 1
                                                lite=True,                # MATLAB: 1
                                                verify=False,             # MATLAB: 0
                                                apply_T_corr=True,        # MATLAB: 1
                                                MoninObukhov=False,       # MATLAB: 0
                                                soil_heat_method=2,       # MATLAB: 2 (G=0.35*Rn)
                                                calc_rss_rbs=False,       # MATLAB: 0
                                            )

                                            # Create atmosphere
                                            atmo = create_simple_atmo(spectral, Rin, meteo.Rli)

                                            # Run SCOPE with pre-loaded optipar
                                            output = run_single_scenario(
                                                leafbio=leafbio,
                                                canopy=canopy,
                                                soil=soil,
                                                meteo=meteo,
                                                angles=angles,
                                                options=options,
                                                spectral=spectral,
                                                optipar=optipar,
                                                atmo=atmo,
                                            )

                                            # Extract fluorescence results
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
                                                # Escape ratio = EoutF_ / Femleaves_
                                                with np.errstate(divide='ignore', invalid='ignore'):
                                                    esc = np.where(fl.Femleaves_ > 0,
                                                                   fl.EoutF_ / fl.Femleaves_, np.nan)
                                                escape_685 = esc[45] if len(esc) > 45 else np.nan
                                                escape_740 = esc[100] if len(esc) > 100 else np.nan
                                                escape_761 = esc[121] if len(esc) > 121 else np.nan
                                            else:
                                                F684 = F685 = F740 = F761 = np.nan
                                                wl685 = wl740 = LoutF = EoutF = np.nan
                                                escape_685 = escape_740 = escape_761 = np.nan

                                            # Extract other results
                                            rad = output['rad']
                                            fluxes = output['fluxes']
                                            thermal = output['thermal']
                                            Eouto = getattr(rad, 'Eouto', np.nan) if rad else np.nan

                                            # Store results
                                            results.append({
                                                'Cab': Cab, 'LAI': LAI, 'LAD': LAD, 'tts': tts,
                                                'tto': tto, 'Rin': Rin, 'Ta': Ta, 'Vcmax': Vcmax,
                                                'soil_type': soil_type,
                                                'F684': F684, 'F685': F685, 'F740': F740,
                                                'F761': F761, 'wl685': wl685, 'wl740': wl740,
                                                'LoutF': LoutF, 'EoutF': EoutF,
                                                'escape_685': escape_685, 'escape_740': escape_740,
                                                'escape_761': escape_761, 'Eouto': Eouto,
                                                'Rntot': fluxes.get('Rntot', np.nan),
                                                'lEtot': fluxes.get('lEtot', np.nan),
                                                'Htot': fluxes.get('Htot', np.nan),
                                                'Actot': fluxes.get('Actot', np.nan),
                                                'Rnctot': fluxes.get('Rnctot', np.nan),
                                                'Tcave': fluxes.get('Tcave', np.nan),
                                                'Tsave': fluxes.get('Tsave', np.nan),
                                                'group': group, 'error': ''
                                            })

                                        except Exception as e:
                                            errors += 1
                                            results.append({
                                                'Cab': Cab, 'LAI': LAI, 'LAD': LAD, 'tts': tts,
                                                'tto': tto, 'Rin': Rin, 'Ta': Ta, 'Vcmax': Vcmax,
                                                'soil_type': soil_type,
                                                'F684': np.nan, 'F685': np.nan, 'F740': np.nan,
                                                'F761': np.nan, 'wl685': np.nan, 'wl740': np.nan,
                                                'LoutF': np.nan, 'EoutF': np.nan,
                                                'escape_685': np.nan, 'escape_740': np.nan,
                                                'escape_761': np.nan,
                                                'Eouto': np.nan, 'Rntot': np.nan, 'lEtot': np.nan,
                                                'Htot': np.nan, 'Actot': np.nan, 'Rnctot': np.nan,
                                                'Tcave': np.nan, 'Tsave': np.nan, 'group': group,
                                                'error': str(e)
                                            })

                                        # Progress
                                        if idx % 500 == 0:
                                            elapsed = time.time() - start_time
                                            rate = idx / elapsed
                                            eta = (n_total - idx) / rate / 60 if rate > 0 else 0
                                            print(f'  {idx}/{n_total} ({100*idx/n_total:.1f}%) - {rate:.1f}/s - ETA: {eta:.1f} min')

                                    if max_scenarios is not None and idx >= max_scenarios:
                                        break
                                if max_scenarios is not None and idx >= max_scenarios:
                                    break
                            if max_scenarios is not None and idx >= max_scenarios:
                                break
                        if max_scenarios is not None and idx >= max_scenarios:
                            break
                    if max_scenarios is not None and idx >= max_scenarios:
                        break
                if max_scenarios is not None and idx >= max_scenarios:
                    break
            if max_scenarios is not None and idx >= max_scenarios:
                break
        if max_scenarios is not None and idx >= max_scenarios:
            break

    # Save results
    elapsed = time.time() - start_time
    print(f'\n=== DONE ===')
    print(f'Total: {idx}, Errors: {errors}, Time: {elapsed/60:.1f} min')

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(output_dir, f'numerical_experiment_python_{timestamp}.csv')
    df.to_csv(outfile, index=False)
    print(f'Saved: {outfile}')

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run SCOPE numerical experiment (simplified)')
    parser.add_argument('--max-scenarios', type=int, default=100,
                        help='Maximum scenarios to run (for testing)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Override the default SCOPE input directory')

    args = parser.parse_args()

    df = run_experiment(
        output_dir=args.output_dir,
        max_scenarios=args.max_scenarios,
        input_dir=args.input_dir,
    )
