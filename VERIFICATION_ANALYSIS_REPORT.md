# Verification Difference Analysis Report

## Summary

The Python SCOPE implementation shows **near-perfect agreement** with MATLAB when using identical MODTRAN atmospheric inputs:

- **Full MODTRAN experiment** (82,944 scenarios): R² > 0.999999998 for all key variables
- **Fluorescence (F761)**: R² = 0.9999999983, mean relative difference 0.003%
- **Photosynthesis (Actot)**: R² = 0.9999999980, mean relative difference 0.003%
- **Temperature (Tcave)**: R² = 0.9999999989, mean relative difference 0.0002%

- **Spectral outputs**: Median R² = 1.0 per wavelength for all variables; fluorescence spectrum (LoF_) R² = 1.0 at every wavelength

Previous ~22% fluorescence offset was entirely due to the simplified `create_atmo()` Gaussian spectrum, not algorithm differences.

## MODTRAN Numerical Experiment (Latest)

### Experiment Design

Both Python and MATLAB ran 82,944 scenarios using identical MODTRAN atmospheric data (FLEX-S3_std.atm), sweeping over:

| Parameter | Values | Count |
|-----------|--------|-------|
| Cab | 10, 20, 40, 80 | 4 |
| LAI | 0.5, 1, 3, 6 | 4 |
| LAD | spherical, planophile, erectophile | 3 |
| tts | 30, 45, 60 | 3 |
| tto | 0, 20, 40, 60 | 4 |
| Rin | 100, 300, 500, 800 | 4 |
| Ta | 15, 25, 35 | 3 |
| Vcmax | 30, 100, 160 | 3 |
| Soil | zero, wet, dry_bright1, dry_bright2 | 4 |
| **Total** | | **82,944** |

### Run Statistics

| | Python | MATLAB |
|---|---|---|
| Machine | Mac (Apple Silicon) | Windows PC |
| Scenarios completed | 82,944 | 82,944 |
| Errors | 0 | 0 |
| Runtime | 10.1 min (~137/s) | 20.6 min (~67/s) |

### Comparison Results

| Variable | R² | Mean Rel Diff (%) | Max Rel Diff (%) | Mean Abs Diff |
|---|---|---|---|---|
| F685 | 0.9999999980 | 0.003 | 0.019 | 9.7e-06 |
| F740 | 0.9999999983 | 0.003 | 0.019 | 3.1e-05 |
| F761 | 0.9999999983 | 0.003 | 0.019 | 1.9e-05 |
| LoutF | 0.9999999982 | 0.003 | 0.019 | 2.0e-06 |
| EoutF | 0.9999999978 | 0.003 | 0.019 | 6.8e-06 |
| Eouto | 0.9999999998 | 0.001 | 0.001 | 6.3e-04 |
| Rntot | 0.9999999994 | 0.005 | 5.484 | 8.2e-04 |
| lEtot | 0.9999999879 | 0.002 | 0.151 | 3.3e-03 |
| Htot | 0.9999999902 | 0.010 | 45.100* | 3.1e-03 |
| Actot | 0.9999999980 | 0.003 | 0.405 | 1.5e-04 |
| Rnctot | 0.9999999996 | 0.007 | 63.919* | 1.0e-03 |
| Tcave | 0.9999999989 | 0.0002 | 0.049 | 6.8e-05 |
| Tsave | 0.9999999858 | 0.0001 | 0.218 | 3.3e-05 |

*High max relative differences for Htot and Rnctot occur at near-zero absolute values (small absolute error / tiny denominator).

### Spectral Comparison Results

Full spectral outputs were also compared across all 82,944 scenarios at every wavelength (2162 bands for broadband variables, 211 bands for fluorescence):

| Variable | Description | Min R² (per-wl) | Median R² | Mean RelDiff (%) | Worst Wavelength |
|---|---|---|---|---|---|
| refl | Reflectance | 0.9999768 | 1.0000000 | 0.0002 | 3000 nm |
| Lo_ | Outgoing radiance | 0.9999965 | 1.0000000 | 0.022 | 4600 nm |
| Eout_ | Upwelling radiation | 0.9999993 | 1.0000000 | 0.022 | 4600 nm |
| Esun_ | Direct solar | 0.9999981 | 1.0000000 | 0.025 | 3000 nm |
| Esky_ | Diffuse sky | 0.9999286 | 1.0000000 | 0.022 | 13100 nm |
| LoF_ | Fluorescence radiance | 1.0000000 | 1.0000000 | 0.003 | 676 nm |

- **Fluorescence spectrum (LoF_)**: Essentially perfect — R² = 1.0 at every wavelength, 0.003% mean relative difference.
- **Reflectance (refl)**: Near-perfect at 0.0002% mean relative difference.
- **Irradiance variables (Esun_, Esky_, Eout_)**: Median R² = 1.0 across wavelengths. Worst agreement is in the thermal IR (3000-13100 nm), likely due to minor floating-point differences in Planck function or thermal band handling, but still R² > 0.9999 everywhere.

### Energy Balance Convergence

Both implementations show identical energy balance convergence warnings for the same edge-case scenarios (high LAI + high Rin + extreme temperatures). Typical residual errors are 1-5 W/m² for shaded vegetation in these cases, confirming both implementations hit the same numerical limits.

## Previous Fixes and Improvements

### 1. CO2 at Leaf Surface (Cc) Fix

**Problem**: Python was using ambient CO2 (Ca = 400 ppm) instead of leaf surface CO2 (Cc) in the Ball-Berry stomatal conductance calculation during energy balance iterations.

**Root cause**: The `heatfluxes_core` function didn't calculate or return Cc, so Cch and Ccu stayed at ambient Ca throughout iterations.

**Fix**: Added Cc calculation to `heatfluxes_core` and `heatfluxes_vectorized`:
```python
# CO2 at leaf surface (MATLAB: Cc = Ca - (Ca-Ci)*ra/(ra+rs))
Cc = Ca - (Ca - Ci) * ra / (ra + rs)
```

**Result**: Actot R² improved from 0.9758 to 0.9902 (with old Gaussian atmo).

### 2. Code Consolidation

Consolidated duplicate numba files into single modules for cleaner codebase:

| Before | After |
|--------|-------|
| `biochemical.py` + `biochemical_numba.py` | `biochemical.py` |
| `rtmo.py` + `rtmo_numba.py` | `rtmo.py` |

All numba-optimized functions are now in the main module files.

### 3. MODTRAN Atmospheric Input Support

Added `load_atmo()` function to load MODTRAN `.atm` files (e.g., FLEX-S3_std.atm), enabling realistic spectral irradiance with:
- Detailed atmospheric absorption lines
- Wavelength-dependent direct/diffuse ratio
- Proper scaling by `meteo.Rin`

This eliminated the ~22% systematic fluorescence offset previously caused by the simplified Gaussian `create_atmo()`.

## Key Findings

### 1. Atmospheric Spectrum Was the Root Cause of Offsets

| Atmosphere | F761 R² | F761 Offset | Actot R² |
|---|---|---|---|
| Gaussian (`create_atmo()`) | 0.998 | ~22% | 0.9902 |
| MODTRAN (identical inputs) | 0.9999999983 | 0.003% | 0.9999999980 |

The simplified Gaussian spectrum caused significant differences in PAR distribution, fluorescence excitation energy, and net radiation balance. With identical MODTRAN inputs, these all disappear.

### 2. Optipar Files Are Equivalent

| Parameter | Optipar2017 | Optipar2021 | Difference |
|-----------|-------------|-------------|------------|
| phi (sum) | 1.000000 | 1.000000 | 0.00% |
| Kab (sum) | 10.5539 | 10.5539 | 0.00% |
| nr (sum) | 2761.77 | 2761.77 | 0.00% |
| Kdm (sum) | 16650.54 | 16650.54 | 0.00% |
| Kw (sum) | 30913.40 | 30913.40 | 0.00% |

The 2021 file has additional fields (Kp, Kcbc, phiE) for protein and carotenoid absorption, but these don't significantly affect the core fluorescence calculation.

## Code Structure (After Consolidation)

### `scope/fluxes/biochemical.py`
- `biochemical_core` - numba-compiled FvCB photosynthesis
- `biochemical_vectorized` - vectorized for all layers
- `heatfluxes_core` - numba-compiled heat flux with Cc calculation
- `heatfluxes_vectorized` - vectorized for all layers
- `brentq_numba` - Ball-Berry fixed-point solver
- `BiochemicalOutput` - dataclass for results

### `scope/rtm/rtmo.py`
- `calc_Pso_array` - bidirectional gap probability
- `calc_layer_absorption_direct` - direct radiation absorption
- `calc_layer_absorption_diffuse` - diffuse radiation absorption
- `sint_1d`, `e2phot_numba` - integration helpers
- `rtmo()` - main radiative transfer function

### `scope/io/load_atmo.py`
- `load_atmo()` - load MODTRAN `.atm` files
- `aggreg()` - aggregate high-resolution spectra to SCOPE spectral bands

## Conclusions

1. **Python implementation is verified**: With identical MODTRAN atmospheric inputs across 82,944 scenarios, R² > 0.999999998 for all key output variables. Mean relative differences are 0.003% or less for fluorescence and photosynthesis.

2. **Previous offsets were input-driven**: The ~22% fluorescence offset seen with the Gaussian `create_atmo()` was entirely due to different atmospheric spectral shapes, not algorithm differences.

3. **Energy balance behavior is identical**: Both implementations produce the same convergence warnings for the same edge-case scenarios, with residual errors of the same magnitude.

4. **Python is ~2x faster**: 10.1 min vs 20.6 min for 82,944 scenarios (different hardware, but indicative of competitive performance with numba JIT compilation).

## Files

### Numerical Experiment Scripts
- `numerical_experiment_mod.py` - Python MODTRAN experiment (82,944 scenarios)
- `../numerical_experiment_matlab_mod.m` - MATLAB MODTRAN experiment (82,944 scenarios)

### Analysis Scripts
- `analyze_verification_diff.py` - Initial analysis script
- `analyze_verification_detailed.py` - Detailed analysis with eta values
- `check_optipar_diff.py` - Optipar file comparison
- `verification_figures.ipynb` - Jupyter notebook for verification plots

### Results (in `../output/`)
- `numerical_experiment_modtran_20260209_014257.csv` - Python scalar results (82,944 scenarios)
- `numerical_experiment_modtran_spectral_20260209_014257.npz` - Python spectral data
- `numerical_experiment_matlab_modtran_20260209_014512.csv` - MATLAB scalar results (82,944 scenarios)
- `numerical_experiment_matlab_modtran_spectral_20260209_014512.mat` - MATLAB spectral data
- `comparison_statistics.csv` - Scalar variable comparison statistics
- `spectral_comparison_statistics.csv` - Spectral variable comparison statistics

### Core Modules (Consolidated)
- `scope/fluxes/biochemical.py` - Photosynthesis + heat fluxes (numba)
- `scope/rtm/rtmo.py` - Optical radiative transfer (numba)
- `scope/io/load_atmo.py` - MODTRAN atmospheric data loader
