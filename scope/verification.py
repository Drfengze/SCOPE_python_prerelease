"""Verification module for SCOPE Python implementation.

Compares Python outputs against MATLAB reference outputs.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .main import run_scope
from .types import LeafBio, Canopy, Soil, Meteo, Angles, Options
from .spectral import SPECTRAL


@dataclass
class VerificationResult:
    """Result of verification comparison."""
    simulation_number: int
    variable: str
    python_value: float
    matlab_value: float
    abs_error: float
    rel_error: float
    passed: bool


def load_matlab_inputs(verification_dir: Path) -> pd.DataFrame:
    """Load input parameters from MATLAB verification data."""
    pars_file = verification_dir / "pars_and_input_short.csv"

    # Read CSV, skip the units row
    df = pd.read_csv(pars_file, skiprows=[1])
    return df


def load_matlab_outputs(verification_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all MATLAB output CSV files."""
    outputs = {}

    csv_files = [
        "fluxes.csv",
        "vegetation.csv",
        "radiation.csv",
        "resistances.csv",
        "aPAR.csv",
        "fluorescence_scalars.csv",
    ]

    for fname in csv_files:
        fpath = verification_dir / fname
        if fpath.exists():
            # Skip units row (row 1)
            df = pd.read_csv(fpath, skiprows=[1])
            outputs[fname.replace(".csv", "")] = df

    return outputs


def create_inputs_from_row(row: pd.Series) -> Tuple[LeafBio, Canopy, Soil, Meteo, Angles]:
    """Create SCOPE input objects from a parameter row."""

    leafbio = LeafBio(
        Cab=row.get('Cab', 40.0),
        Cca=row.get('Cca', 10.0),
        Cdm=row.get('Cdm', 0.003),
        Cw=row.get('Cw', 0.01),
        Cs=row.get('Cs', 0.0),
        N=row.get('N', 1.5),
        Cant=row.get('Cant', 0.0),
        Vcmax25=row.get('Vcmax25', 60.0),
        BallBerrySlope=row.get('BallBerrySlope', 8.0),
    )

    canopy = Canopy(
        LAI=row.get('LAI', 3.0),
        LIDFa=row.get('LIDFa', -0.35),
        LIDFb=row.get('LIDFb', -0.15),
        hc=row.get('hc', 2.0),
    )

    soil = Soil()

    meteo = Meteo(
        Rin=row.get('Rin', 600.0),
        Rli=row.get('Rli', 300.0),  # MATLAB verification uses constant Rli=300 W/m²
        Ta=row.get('Ta', 20.0),
        ea=row.get('ea', 15.0),
        p=row.get('p', 1013.0),
        u=row.get('u', 2.0),
    )

    angles = Angles(
        tts=row.get('tts', 30.0),
        tto=row.get('tto', 0.0),
        psi=row.get('psi', 0.0),
    )

    return leafbio, canopy, soil, meteo, angles


def run_verification(
    verification_dir: str = None,
    output_dir: str = None,
    n_simulations: int = None,
    tolerance: float = 0.1,
) -> Tuple[List[VerificationResult], Dict[str, np.ndarray]]:
    """Run verification against MATLAB outputs.

    Args:
        verification_dir: Path to MATLAB verification data
        output_dir: Path to save Python outputs (optional)
        n_simulations: Number of simulations to run (None = all)
        tolerance: Relative tolerance for comparison

    Returns:
        Tuple of (list of VerificationResult, dict of Python outputs)
    """
    # Default paths
    if verification_dir is None:
        # Go up from scope/verification.py to SCOPE-master/output/verificationdata
        verification_dir = Path(__file__).parent.parent.parent / "output" / "verificationdata"
    else:
        verification_dir = Path(verification_dir)

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output" / "verification"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MATLAB data
    print(f"Loading MATLAB verification data from: {verification_dir}")
    matlab_inputs = load_matlab_inputs(verification_dir)
    matlab_outputs = load_matlab_outputs(verification_dir)

    if n_simulations is None:
        n_simulations = len(matlab_inputs)
    else:
        n_simulations = min(n_simulations, len(matlab_inputs))

    print(f"Running {n_simulations} verification simulations...")

    # Storage for Python outputs
    python_outputs = {
        'fluxes': [],
        'vegetation': [],
        'radiation': [],
        'resistances': [],
        'aPAR': [],
    }

    results = []

    for i in range(n_simulations):
        row = matlab_inputs.iloc[i]
        sim_num = int(row.get('simulation_number', i + 1)) if 'simulation_number' in row else i + 1

        print(f"  Simulation {sim_num}...", end=" ")

        try:
            # Create inputs
            leafbio, canopy, soil, meteo, angles = create_inputs_from_row(row)

            # Run Python SCOPE
            output = run_scope(leafbio, canopy, soil, meteo, angles)

            # Extract outputs in MATLAB format
            fluxes_row = {
                'simulation_number': sim_num,
                'nu_iterations': output.iter_count,
                'year': 0,
                'DoY': 0,
                'Rnctot': output.fluxes.get('Rnctot', 0),
                'lEctot': output.fluxes.get('lEctot', 0),
                'Hctot': output.fluxes.get('Hctot', 0),
                'Actot': output.fluxes.get('Actot', 0),
                'Tcave': output.fluxes.get('Tcave', 0),
                'Rnstot': output.fluxes.get('Rnstot', 0),
                'lEstot': output.fluxes.get('lEstot', 0),
                'Hstot': output.fluxes.get('Hstot', 0),
                'Gtot': output.fluxes.get('Gtot', 0),
                'Tsave': output.fluxes.get('Tsave', 0),
                'Rntot': output.fluxes.get('Rntot', 0),
                'lEtot': output.fluxes.get('lEtot', 0),
                'Htot': output.fluxes.get('Htot', 0),
            }
            python_outputs['fluxes'].append(fluxes_row)

            vegetation_row = {
                'simulation_number': sim_num,
                'year': 0,
                'DoY': 0,
                'Photosynthesis': output.canopy.A,
                'Electron_transport': output.canopy.Ja,
                'NPQ_energy': 0,  # TODO: compute
                'NPQ_photon': 0,  # TODO: compute
                'canopy_level_FQE': 0,  # TODO: compute
                'LST': output.canopy.LST,
                'emis': output.canopy.emis,
                'GPP': output.canopy.GPP,
            }
            python_outputs['vegetation'].append(vegetation_row)

            aPAR_row = {
                'simulation_number': sim_num,
                'year': 0,
                'DoY': 0,
                'iPAR': output.rad.PAR,
                'LAIsunlit': output.canopy.LAIsunlit,
                'LAIshaded': output.canopy.LAIshaded,
                'aPARtot': output.canopy.Pntot,
                'aPARsun': output.canopy.Pnsun,
                'aPARsha': output.canopy.Pnsha,
                'aPARCabtot': output.canopy.Pntot_Cab,
                'aPARCabsun': output.canopy.Pnsun_Cab,
                'aPARCabsha': output.canopy.Pnsha_Cab,
            }
            python_outputs['aPAR'].append(aPAR_row)

            print("OK")

            # Compare key variables
            if 'vegetation' in matlab_outputs:
                matlab_veg = matlab_outputs['vegetation']
                if i < len(matlab_veg):
                    matlab_row = matlab_veg.iloc[i]

                    comparisons = [
                        ('GPP', vegetation_row['GPP'], matlab_row.get('GPP', 0)),
                        ('Photosynthesis', vegetation_row['Photosynthesis'], matlab_row.get('Photosynthesis', 0)),
                        ('Electron_transport', vegetation_row['Electron_transport'], matlab_row.get('Electron_transport', 0)),
                        ('LST', vegetation_row['LST'], matlab_row.get('LST', 0)),
                    ]

                    for var_name, py_val, mat_val in comparisons:
                        if mat_val != 0:
                            rel_err = abs(py_val - mat_val) / abs(mat_val)
                        else:
                            rel_err = abs(py_val) if py_val != 0 else 0

                        results.append(VerificationResult(
                            simulation_number=sim_num,
                            variable=var_name,
                            python_value=py_val,
                            matlab_value=mat_val,
                            abs_error=abs(py_val - mat_val),
                            rel_error=rel_err,
                            passed=rel_err <= tolerance,
                        ))

            if 'fluxes' in matlab_outputs:
                matlab_flux = matlab_outputs['fluxes']
                if i < len(matlab_flux):
                    matlab_row = matlab_flux.iloc[i]

                    comparisons = [
                        ('Rntot', fluxes_row['Rntot'], matlab_row.get('Rntot', 0)),
                        ('lEtot', fluxes_row['lEtot'], matlab_row.get('lEtot', 0)),
                        ('Htot', fluxes_row['Htot'], matlab_row.get('Htot', 0)),
                        ('Actot', fluxes_row['Actot'], matlab_row.get('Actot', 0)),
                    ]

                    for var_name, py_val, mat_val in comparisons:
                        if mat_val != 0:
                            rel_err = abs(py_val - mat_val) / abs(mat_val)
                        else:
                            rel_err = abs(py_val) if py_val != 0 else 0

                        results.append(VerificationResult(
                            simulation_number=sim_num,
                            variable=var_name,
                            python_value=py_val,
                            matlab_value=mat_val,
                            abs_error=abs(py_val - mat_val),
                            rel_error=rel_err,
                            passed=rel_err <= tolerance,
                        ))

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Save Python outputs
    for name, data in python_outputs.items():
        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_dir / f"{name}.csv", index=False)
            np.save(output_dir / f"{name}.npy", df.to_numpy())

    print(f"\nPython outputs saved to: {output_dir}")

    return results, python_outputs


def print_verification_report(results: List[VerificationResult]):
    """Print a summary report of verification results."""
    if not results:
        print("No verification results to report.")
        return

    # Group by variable
    variables = {}
    for r in results:
        if r.variable not in variables:
            variables[r.variable] = []
        variables[r.variable].append(r)

    print("\n" + "=" * 80)
    print("VERIFICATION REPORT")
    print("=" * 80)

    total_passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\nOverall: {total_passed}/{total} comparisons passed ({100*total_passed/total:.1f}%)")
    print()

    for var_name, var_results in sorted(variables.items()):
        passed = sum(1 for r in var_results if r.passed)
        total_var = len(var_results)

        rel_errors = [r.rel_error for r in var_results]
        mean_err = np.mean(rel_errors)
        max_err = np.max(rel_errors)

        status = "✓" if passed == total_var else "✗"
        print(f"{status} {var_name:25s}: {passed}/{total_var} passed, "
              f"mean rel.err={mean_err:.2%}, max={max_err:.2%}")

    # Show worst cases
    print("\nWorst mismatches (top 5):")
    sorted_results = sorted(results, key=lambda r: r.rel_error, reverse=True)
    for r in sorted_results[:5]:
        print(f"  Sim {r.simulation_number}, {r.variable}: "
              f"Python={r.python_value:.4e}, MATLAB={r.matlab_value:.4e}, "
              f"rel.err={r.rel_error:.2%}")


if __name__ == "__main__":
    results, outputs = run_verification(n_simulations=10)
    print_verification_report(results)
