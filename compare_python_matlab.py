#!/usr/bin/env python3
"""
Compare Python and MATLAB numerical experiment results.

Includes both scalar and spectral output comparisons.

Usage:
    python compare_python_matlab.py <python_csv> <matlab_csv> [--spectral-py <npz>] [--spectral-mat <mat>]
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
import matplotlib.pyplot as plt


def load_and_align_data(python_file: str, matlab_file: str) -> tuple:
    """Load both CSV files and align by parameter combinations."""

    print("Loading data...")
    df_py = pd.read_csv(python_file)
    df_mat = pd.read_csv(matlab_file)

    print(f"  Python: {len(df_py)} rows")
    print(f"  MATLAB: {len(df_mat)} rows")

    # Define key columns for matching
    key_cols = ['Cab', 'LAI', 'LAD', 'tts', 'tto', 'Rin', 'Ta', 'Vcmax', 'soil_type']

    # Merge on key columns
    df_merged = pd.merge(
        df_py, df_mat,
        on=key_cols,
        suffixes=('_py', '_mat'),
        how='inner'
    )

    print(f"  Matched: {len(df_merged)} rows")

    return df_py, df_mat, df_merged


def load_spectral_data(python_npz: str, matlab_mat: str) -> tuple:
    """Load spectral data from Python NPZ and MATLAB MAT files.

    Args:
        python_npz: Path to Python .npz spectral file
        matlab_mat: Path to MATLAB .mat spectral file

    Returns:
        Tuple of (py_spec, mat_spec) dictionaries
    """
    print("\nLoading spectral data...")

    # Load Python NPZ
    py_data = np.load(python_npz)
    py_spec = {
        'wl': py_data['wl'],
        'wlF': py_data['wlF'],
        'refl': py_data['refl'],
        'Lo_': py_data['Lo_'],
        'Eout_': py_data['Eout_'],
        'Esun_': py_data['Esun_'],
        'Esky_': py_data['Esky_'],
        'LoF_': py_data['LoF_'],
    }
    print(f"  Python spectral: {py_spec['refl'].shape[0]} scenarios x {py_spec['refl'].shape[1]} wavelengths")

    # Load MATLAB MAT - try scipy first, then h5py for v7.3 files
    try:
        mat_data = sio.loadmat(matlab_mat)
        spectral_data = mat_data['spectral_data']
        # Extract from MATLAB struct (scipy format)
        mat_spec = {
            'wl': spectral_data['wl'][0, 0].flatten(),
            'wlF': spectral_data['wlF'][0, 0].flatten(),
            'refl': spectral_data['refl'][0, 0],
            'Lo_': spectral_data['Lo_'][0, 0],
            'Eout_': spectral_data['Eout_'][0, 0],
            'Esun_': spectral_data['Esun_'][0, 0],
            'Esky_': spectral_data['Esky_'][0, 0],
            'LoF_': spectral_data['LoF_'][0, 0],
        }
    except NotImplementedError:
        # MATLAB v7.3 file - use h5py
        import h5py
        print("  Using h5py for MATLAB v7.3 file...")
        with h5py.File(matlab_mat, 'r') as f:
            spectral_data = f['spectral_data']
            mat_spec = {
                'wl': np.array(spectral_data['wl']).flatten(),
                'wlF': np.array(spectral_data['wlF']).flatten(),
                'refl': np.array(spectral_data['refl']).T,  # Transpose for h5py
                'Lo_': np.array(spectral_data['Lo_']).T,
                'Eout_': np.array(spectral_data['Eout_']).T,
                'Esun_': np.array(spectral_data['Esun_']).T,
                'Esky_': np.array(spectral_data['Esky_']).T,
                'LoF_': np.array(spectral_data['LoF_']).T,
            }
    print(f"  MATLAB spectral: {mat_spec['refl'].shape[0]} scenarios x {mat_spec['refl'].shape[1]} wavelengths")

    return py_spec, mat_spec


def compare_spectral_at_wavelengths(py_spec: dict, mat_spec: dict, target_wls: list = None) -> pd.DataFrame:
    """Compare spectral data at specific wavelengths.

    Args:
        py_spec: Python spectral data dictionary
        mat_spec: MATLAB spectral data dictionary
        target_wls: List of target wavelengths (nm). If None, uses defaults.

    Returns:
        DataFrame with comparison statistics at each wavelength
    """
    if target_wls is None:
        # Key wavelengths for vegetation studies
        target_wls = [400, 450, 500, 550, 600, 650, 680, 700, 740, 760, 800, 850, 900, 1000, 1500, 2000]

    wl = py_spec['wl']
    results = []

    for var_name in ['refl', 'Lo_', 'Eout_', 'Esun_', 'Esky_']:
        py_arr = py_spec[var_name]
        mat_arr = mat_spec[var_name]

        for target in target_wls:
            # Find closest wavelength index
            idx = np.argmin(np.abs(wl - target))
            actual_wl = wl[idx]

            py_vals = py_arr[:, idx]
            mat_vals = mat_arr[:, idx]

            # Remove NaN pairs
            valid = ~(np.isnan(py_vals) | np.isnan(mat_vals))
            py_valid = py_vals[valid]
            mat_valid = mat_vals[valid]

            if len(py_valid) == 0:
                continue

            # Statistics
            diff = py_valid - mat_valid
            abs_diff = np.abs(diff)
            denom = np.maximum(np.abs(mat_valid), 1e-10)
            rel_diff = abs_diff / denom * 100

            # Correlation and R²
            if np.std(py_valid) > 0 and np.std(mat_valid) > 0:
                corr = np.corrcoef(py_valid, mat_valid)[0, 1]
                ss_res = np.sum((py_valid - mat_valid)**2)
                ss_tot = np.sum((mat_valid - np.mean(mat_valid))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            else:
                corr, r2 = np.nan, np.nan

            results.append({
                'Variable': var_name,
                'Wavelength_nm': actual_wl,
                'N': len(py_valid),
                'Mean_Python': np.mean(py_valid),
                'Mean_MATLAB': np.mean(mat_valid),
                'Mean_Diff': np.mean(diff),
                'Mean_AbsDiff': np.mean(abs_diff),
                'Mean_RelDiff_%': np.mean(rel_diff),
                'Max_RelDiff_%': np.max(rel_diff),
                'Correlation': corr,
                'R_squared': r2,
            })

    return pd.DataFrame(results)


def compare_fluorescence_spectrum(py_spec: dict, mat_spec: dict) -> pd.DataFrame:
    """Compare fluorescence spectrum (LoF_).

    Args:
        py_spec: Python spectral data dictionary
        mat_spec: MATLAB spectral data dictionary

    Returns:
        DataFrame with comparison statistics at each fluorescence wavelength
    """
    wlF = py_spec['wlF']
    py_arr = py_spec['LoF_']
    mat_arr = mat_spec['LoF_']

    results = []

    # Sample wavelengths for fluorescence
    target_wls = [640, 660, 680, 685, 700, 720, 740, 760, 780, 800, 850]

    for target in target_wls:
        idx = np.argmin(np.abs(wlF - target))
        actual_wl = wlF[idx]

        py_vals = py_arr[:, idx]
        mat_vals = mat_arr[:, idx]

        # Remove NaN pairs
        valid = ~(np.isnan(py_vals) | np.isnan(mat_vals))
        py_valid = py_vals[valid]
        mat_valid = mat_vals[valid]

        if len(py_valid) == 0:
            continue

        # Statistics
        diff = py_valid - mat_valid
        abs_diff = np.abs(diff)
        denom = np.maximum(np.abs(mat_valid), 1e-10)
        rel_diff = abs_diff / denom * 100

        if np.std(py_valid) > 0 and np.std(mat_valid) > 0:
            corr = np.corrcoef(py_valid, mat_valid)[0, 1]
            ss_res = np.sum((py_valid - mat_valid)**2)
            ss_tot = np.sum((mat_valid - np.mean(mat_valid))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            corr, r2 = np.nan, np.nan

        results.append({
            'Variable': 'LoF_',
            'Wavelength_nm': actual_wl,
            'N': len(py_valid),
            'Mean_Python': np.mean(py_valid),
            'Mean_MATLAB': np.mean(mat_valid),
            'Mean_Diff': np.mean(diff),
            'Mean_AbsDiff': np.mean(abs_diff),
            'Mean_RelDiff_%': np.mean(rel_diff),
            'Max_RelDiff_%': np.max(rel_diff),
            'Correlation': corr,
            'R_squared': r2,
        })

    return pd.DataFrame(results)


def compute_spectral_rmse_per_scenario(py_spec: dict, mat_spec: dict, var_name: str = 'refl') -> np.ndarray:
    """Compute RMSE between Python and MATLAB spectra for each scenario.

    Args:
        py_spec: Python spectral data
        mat_spec: MATLAB spectral data
        var_name: Variable to compare ('refl', 'Lo_', etc.)

    Returns:
        Array of RMSE values for each scenario
    """
    py_arr = py_spec[var_name]
    mat_arr = mat_spec[var_name]

    n_scenarios = py_arr.shape[0]
    rmse = np.zeros(n_scenarios)

    for i in range(n_scenarios):
        py_row = py_arr[i, :]
        mat_row = mat_arr[i, :]

        valid = ~(np.isnan(py_row) | np.isnan(mat_row))
        if np.sum(valid) > 0:
            rmse[i] = np.sqrt(np.mean((py_row[valid] - mat_row[valid])**2))
        else:
            rmse[i] = np.nan

    return rmse


def plot_spectral_comparison(py_spec: dict, mat_spec: dict, scenario_idx: int,
                             output_path: str = None):
    """Plot spectral comparison for a single scenario.

    Args:
        py_spec: Python spectral data
        mat_spec: MATLAB spectral data
        scenario_idx: Index of scenario to plot
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    wl = py_spec['wl']
    wlF = py_spec['wlF']

    # Reflectance
    ax = axes[0, 0]
    ax.plot(wl, py_spec['refl'][scenario_idx, :], 'b-', label='Python', alpha=0.7)
    ax.plot(wl, mat_spec['refl'][scenario_idx, :], 'r--', label='MATLAB', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title('Reflectance')
    ax.legend()
    ax.set_xlim([400, 2500])

    # Outgoing radiance
    ax = axes[0, 1]
    ax.plot(wl, py_spec['Lo_'][scenario_idx, :], 'b-', label='Python', alpha=0.7)
    ax.plot(wl, mat_spec['Lo_'][scenario_idx, :], 'r--', label='MATLAB', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (mW/m²/nm/sr)')
    ax.set_title('Outgoing Radiance (Lo_)')
    ax.legend()
    ax.set_xlim([400, 2500])

    # Esun
    ax = axes[0, 2]
    ax.plot(wl, py_spec['Esun_'][scenario_idx, :], 'b-', label='Python', alpha=0.7)
    ax.plot(wl, mat_spec['Esun_'][scenario_idx, :], 'r--', label='MATLAB', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Irradiance (mW/m²/nm)')
    ax.set_title('Direct Solar (Esun_)')
    ax.legend()
    ax.set_xlim([400, 2500])

    # Esky
    ax = axes[1, 0]
    ax.plot(wl, py_spec['Esky_'][scenario_idx, :], 'b-', label='Python', alpha=0.7)
    ax.plot(wl, mat_spec['Esky_'][scenario_idx, :], 'r--', label='MATLAB', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Irradiance (mW/m²/nm)')
    ax.set_title('Diffuse Sky (Esky_)')
    ax.legend()
    ax.set_xlim([400, 2500])

    # Fluorescence spectrum
    ax = axes[1, 1]
    ax.plot(wlF, py_spec['LoF_'][scenario_idx, :], 'b-', label='Python', alpha=0.7)
    ax.plot(wlF, mat_spec['LoF_'][scenario_idx, :], 'r--', label='MATLAB', alpha=0.7)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Fluorescence (mW/m²/nm/sr)')
    ax.set_title('Fluorescence Radiance (LoF_)')
    ax.legend()

    # Difference in reflectance
    ax = axes[1, 2]
    diff = py_spec['refl'][scenario_idx, :] - mat_spec['refl'][scenario_idx, :]
    ax.plot(wl, diff, 'g-', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Difference')
    ax.set_title('Reflectance Difference (Python - MATLAB)')
    ax.set_xlim([400, 2500])

    plt.suptitle(f'Spectral Comparison - Scenario {scenario_idx}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectral plot: {output_path}")

    plt.close()


def plot_spectral_statistics(spectral_stats: pd.DataFrame, output_path: str = None):
    """Plot spectral comparison statistics.

    Args:
        spectral_stats: DataFrame from compare_spectral_at_wavelengths
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # R² by wavelength for reflectance
    ax = axes[0, 0]
    refl_stats = spectral_stats[spectral_stats['Variable'] == 'refl']
    ax.plot(refl_stats['Wavelength_nm'], refl_stats['R_squared'], 'b-o', markersize=4)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('R²')
    ax.set_title('Reflectance R² by Wavelength')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # Mean relative difference by wavelength for reflectance
    ax = axes[0, 1]
    ax.plot(refl_stats['Wavelength_nm'], refl_stats['Mean_RelDiff_%'], 'r-o', markersize=4)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Mean Relative Difference (%)')
    ax.set_title('Reflectance Mean Rel. Diff. by Wavelength')
    ax.grid(True, alpha=0.3)

    # R² by wavelength for Lo_
    ax = axes[1, 0]
    lo_stats = spectral_stats[spectral_stats['Variable'] == 'Lo_']
    ax.plot(lo_stats['Wavelength_nm'], lo_stats['R_squared'], 'b-o', markersize=4)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('R²')
    ax.set_title('Outgoing Radiance (Lo_) R² by Wavelength')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # R² by wavelength for Esun_ and Esky_
    ax = axes[1, 1]
    esun_stats = spectral_stats[spectral_stats['Variable'] == 'Esun_']
    esky_stats = spectral_stats[spectral_stats['Variable'] == 'Esky_']
    ax.plot(esun_stats['Wavelength_nm'], esun_stats['R_squared'], 'b-o', markersize=4, label='Esun_')
    ax.plot(esky_stats['Wavelength_nm'], esky_stats['R_squared'], 'r-s', markersize=4, label='Esky_')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('R²')
    ax.set_title('Irradiance R² by Wavelength')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved spectral statistics plot: {output_path}")

    plt.close()


def compute_statistics(df_merged: pd.DataFrame, output_cols: list) -> pd.DataFrame:
    """Compute comparison statistics for output columns."""

    stats = []

    for col in output_cols:
        col_py = f'{col}_py'
        col_mat = f'{col}_mat'

        if col_py not in df_merged.columns or col_mat not in df_merged.columns:
            continue

        py_vals = df_merged[col_py].values
        mat_vals = df_merged[col_mat].values

        # Remove NaN pairs
        valid = ~(np.isnan(py_vals) | np.isnan(mat_vals))
        py_valid = py_vals[valid]
        mat_valid = mat_vals[valid]

        if len(py_valid) == 0:
            continue

        # Absolute difference
        diff = py_valid - mat_valid
        abs_diff = np.abs(diff)

        # Relative difference (avoid division by zero)
        denom = np.maximum(np.abs(mat_valid), 1e-10)
        rel_diff = abs_diff / denom * 100

        # Correlation
        if np.std(py_valid) > 0 and np.std(mat_valid) > 0:
            corr = np.corrcoef(py_valid, mat_valid)[0, 1]
        else:
            corr = np.nan

        # R-squared
        ss_res = np.sum((py_valid - mat_valid)**2)
        ss_tot = np.sum((mat_valid - np.mean(mat_valid))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        stats.append({
            'Variable': col,
            'N': len(py_valid),
            'Mean_Python': np.mean(py_valid),
            'Mean_MATLAB': np.mean(mat_valid),
            'Mean_Diff': np.mean(diff),
            'Std_Diff': np.std(diff),
            'Mean_AbsDiff': np.mean(abs_diff),
            'Max_AbsDiff': np.max(abs_diff),
            'Mean_RelDiff_%': np.mean(rel_diff),
            'Max_RelDiff_%': np.max(rel_diff),
            'Correlation': corr,
            'R_squared': r2,
        })

    return pd.DataFrame(stats)


def analyze_by_group(df_merged: pd.DataFrame, output_cols: list) -> dict:
    """Analyze differences by soil type group."""

    results = {}

    # By soil type
    for soil_type in df_merged['soil_type'].unique():
        subset = df_merged[df_merged['soil_type'] == soil_type]
        stats = compute_statistics(subset, output_cols)
        results[f'soil_{soil_type}'] = stats

    # By LAI
    for lai in df_merged['LAI'].unique():
        subset = df_merged[df_merged['LAI'] == lai]
        stats = compute_statistics(subset, output_cols)
        results[f'LAI_{lai}'] = stats

    return results


def identify_outliers(df_merged: pd.DataFrame, col: str, threshold_pct: float = 10) -> pd.DataFrame:
    """Identify scenarios with large differences."""

    col_py = f'{col}_py'
    col_mat = f'{col}_mat'

    if col_py not in df_merged.columns or col_mat not in df_merged.columns:
        return pd.DataFrame()

    df = df_merged.copy()
    df['diff'] = df[col_py] - df[col_mat]
    df['rel_diff_pct'] = np.abs(df['diff']) / np.maximum(np.abs(df[col_mat]), 1e-10) * 100

    outliers = df[df['rel_diff_pct'] > threshold_pct]

    return outliers[['Cab', 'LAI', 'LAD', 'tts', 'tto', 'Rin', 'Ta', 'Vcmax',
                     'soil_type', col_py, col_mat, 'diff', 'rel_diff_pct']]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare Python and MATLAB numerical experiment results')
    parser.add_argument('python_csv', help='Path to Python CSV results')
    parser.add_argument('matlab_csv', help='Path to MATLAB CSV results')
    parser.add_argument('--spectral-py', help='Path to Python spectral NPZ file')
    parser.add_argument('--spectral-mat', help='Path to MATLAB spectral MAT file')
    parser.add_argument('--plot-scenarios', type=int, nargs='+', default=[0, 1000, 5000],
                        help='Scenario indices to plot (default: 0 1000 5000)')

    args = parser.parse_args()

    python_file = args.python_csv
    matlab_file = args.matlab_csv

    # Load and align scalar data
    df_py, df_mat, df_merged = load_and_align_data(python_file, matlab_file)

    # Output columns to compare
    output_cols = ['F685', 'F740', 'F761', 'LoutF', 'EoutF', 'Eouto',
                   'Rntot', 'lEtot', 'Htot', 'Actot', 'Rnctot', 'Tcave', 'Tsave']

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON STATISTICS (SCALAR OUTPUTS)")
    print("=" * 80)

    stats = compute_statistics(df_merged, output_cols)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(stats.to_string(index=False))

    # Key metrics summary
    print("\n" + "=" * 80)
    print("KEY METRICS SUMMARY")
    print("=" * 80)

    key_vars = ['F740', 'Actot', 'Rntot', 'lEtot', 'Htot']
    for var in key_vars:
        row = stats[stats['Variable'] == var]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"\n{var}:")
            print(f"  Mean relative difference: {r['Mean_RelDiff_%']:.2f}%")
            print(f"  Max relative difference:  {r['Max_RelDiff_%']:.2f}%")
            print(f"  Correlation: {r['Correlation']:.6f}")
            print(f"  R²: {r['R_squared']:.6f}")

    # Identify outliers for F740
    print("\n" + "=" * 80)
    print("OUTLIERS (>10% difference) for F740")
    print("=" * 80)

    outliers = identify_outliers(df_merged, 'F740', threshold_pct=10)
    if len(outliers) > 0:
        print(f"Found {len(outliers)} outliers:")
        print(outliers.head(20).to_string(index=False))
    else:
        print("No outliers found (all differences < 10%)")

    # By soil type
    print("\n" + "=" * 80)
    print("COMPARISON BY SOIL TYPE")
    print("=" * 80)

    for soil_type in df_merged['soil_type'].unique():
        subset = df_merged[df_merged['soil_type'] == soil_type]
        sub_stats = compute_statistics(subset, ['F740', 'Actot'])

        f740_row = sub_stats[sub_stats['Variable'] == 'F740']
        if len(f740_row) > 0:
            r = f740_row.iloc[0]
            print(f"\n{soil_type}: F740 mean rel diff = {r['Mean_RelDiff_%']:.2f}%, R² = {r['R_squared']:.4f}")

    # Save scalar results
    output_dir = Path(python_file).parent
    stats.to_csv(output_dir / 'comparison_statistics.csv', index=False)
    print(f"\nDetailed statistics saved to: {output_dir / 'comparison_statistics.csv'}")

    # ========================================================================
    # SPECTRAL COMPARISON
    # ========================================================================
    if args.spectral_py and args.spectral_mat:
        print("\n" + "=" * 80)
        print("SPECTRAL COMPARISON")
        print("=" * 80)

        py_spec, mat_spec = load_spectral_data(args.spectral_py, args.spectral_mat)

        # Compare at key wavelengths
        print("\n--- Spectral comparison at key wavelengths ---")
        spectral_stats = compare_spectral_at_wavelengths(py_spec, mat_spec)
        print(spectral_stats.to_string(index=False))

        # Save spectral statistics
        spectral_stats.to_csv(output_dir / 'comparison_spectral_statistics.csv', index=False)
        print(f"\nSpectral statistics saved to: {output_dir / 'comparison_spectral_statistics.csv'}")

        # Fluorescence spectrum comparison
        print("\n--- Fluorescence spectrum comparison ---")
        fluor_stats = compare_fluorescence_spectrum(py_spec, mat_spec)
        print(fluor_stats.to_string(index=False))

        fluor_stats.to_csv(output_dir / 'comparison_fluorescence_statistics.csv', index=False)
        print(f"Fluorescence statistics saved to: {output_dir / 'comparison_fluorescence_statistics.csv'}")

        # Compute RMSE per scenario for reflectance
        print("\n--- RMSE per scenario statistics ---")
        rmse_refl = compute_spectral_rmse_per_scenario(py_spec, mat_spec, 'refl')
        print(f"Reflectance RMSE: mean={np.nanmean(rmse_refl):.6f}, max={np.nanmax(rmse_refl):.6f}")

        rmse_lo = compute_spectral_rmse_per_scenario(py_spec, mat_spec, 'Lo_')
        print(f"Lo_ RMSE: mean={np.nanmean(rmse_lo):.4f}, max={np.nanmax(rmse_lo):.4f}")

        rmse_lof = compute_spectral_rmse_per_scenario(py_spec, mat_spec, 'LoF_')
        print(f"LoF_ RMSE: mean={np.nanmean(rmse_lof):.6f}, max={np.nanmax(rmse_lof):.6f}")

        # Plot spectral statistics
        plot_spectral_statistics(spectral_stats, output_dir / 'spectral_statistics_plot.png')

        # Plot example scenarios
        print("\n--- Plotting example scenarios ---")
        for idx in args.plot_scenarios:
            if idx < py_spec['refl'].shape[0]:
                plot_spectral_comparison(py_spec, mat_spec, idx,
                                        output_dir / f'spectral_comparison_scenario_{idx}.png')

        # Summary of spectral comparison
        print("\n" + "=" * 80)
        print("SPECTRAL COMPARISON SUMMARY")
        print("=" * 80)

        # Get overall R² for key variables at key wavelengths
        for var in ['refl', 'Lo_', 'Esun_', 'Esky_']:
            var_stats = spectral_stats[spectral_stats['Variable'] == var]
            if len(var_stats) > 0:
                mean_r2 = var_stats['R_squared'].mean()
                min_r2 = var_stats['R_squared'].min()
                mean_rel = var_stats['Mean_RelDiff_%'].mean()
                print(f"{var}: Mean R²={mean_r2:.4f}, Min R²={min_r2:.4f}, Mean Rel Diff={mean_rel:.2f}%")

        if len(fluor_stats) > 0:
            mean_r2 = fluor_stats['R_squared'].mean()
            min_r2 = fluor_stats['R_squared'].min()
            mean_rel = fluor_stats['Mean_RelDiff_%'].mean()
            print(f"LoF_: Mean R²={mean_r2:.4f}, Min R²={min_r2:.4f}, Mean Rel Diff={mean_rel:.2f}%")

    else:
        print("\nNote: No spectral files provided. Use --spectral-py and --spectral-mat to compare spectra.")


if __name__ == '__main__':
    main()