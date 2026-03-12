#!/usr/bin/env python3
"""
SCOPE.py - Python equivalent of SCOPE.m

SCOPE is a coupled radiative transfer and energy balance model.
Option 'lite' runs a computationally lighter variation of the model,
with the net radiation and leaf temperatures of leaf inclination classes averaged.

This Python implementation follows the structure of SCOPE.m exactly.

Copyright (C) 2021  Christiaan van der Tol (original MATLAB)
Python translation: 2024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.
"""

import sys
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import SCOPE modules
from scope.spectral import SpectralBands
from scope.constants import CONSTANTS
from scope.types import LeafBio, Canopy, Soil, Meteo, Angles, Options
from scope.io.load_optipar import load_optipar
from scope.rtm.fluspect import fluspect
from scope.rtm.rtmo import rtmo
from scope.rtm.rtmf import rtmf
from scope.fluxes.ebal import ebal
from scope.supporting.leafangles import compute_canopy_lidf


class SCOPEModel:
    """
    SCOPE Model - Python implementation following MATLAB SCOPE.m structure.

    This class encapsulates the SCOPE model workflow:
    1. Define constants
    2. Load input data
    3. Load spectral data for leaf and soil
    4. Define canopy structure and fixed parameters
    5. Define spectral regions
    6. Run the radiative transfer and energy balance models
    7. Compute output products
    """

    def __init__(self, input_dir: Optional[str] = None):
        """
        Initialize SCOPE model.

        Args:
            input_dir: Path to input directory (default: ../input relative to this file)
        """
        # Set paths
        self.script_dir = Path(__file__).parent
        self.scope_dir = self.script_dir

        if input_dir is None:
            self.input_dir = self.scope_dir / 'input'
        else:
            self.input_dir = Path(input_dir)

        # 1. Define constants (SCOPE.m line 37)
        self.constants = CONSTANTS

        # 2. Initialize spectral bands (SCOPE.m line 195)
        self.spectral = SpectralBands()

        # 3. Load optipar (SCOPE.m line 177)
        self.optipar, self.wlP = load_optipar()

        # 4. Define fixed canopy parameters (SCOPE.m lines 183-188)
        self.canopy_fixed = {
            'nlincl': 13,
            'nlazi': 36,
            'litab': np.array([5, 15, 25, 35, 45, 55, 65, 75, 81, 83, 85, 87, 89], dtype=float),
            'lazitab': np.arange(5, 356, 10, dtype=float),
        }

        # Soil empirical parameters (SCOPE.m lines 187-188)
        self.soilemp = {
            'SMC': 25,      # empirical parameter (fixed) for BSM
            'film': 0.015,  # empirical parameter (fixed) for BSM
        }

        # Default options (matching setoptions.csv defaults)
        self.default_options = {
            'lite': True,                    # Use computationally lighter SCOPE_lite
            'calc_fluor': True,              # Calculate chlorophyll fluorescence
            'calc_planck': True,             # Calculate thermal radiation spectrum
            'calc_xanthophyll': False,       # Include xanthophyll dependence
            'soilspectrum': 0,               # 0: use soil reflectance from file
            'Fluorescence_model': 0,         # 0: empirical with sustained NPQ
            'apply_T_corr': True,            # Temperature correction for Vcmax
            'verify': False,
            'saveCSV': True,
            'mSCOPE': False,
            'simulation': 0,                 # 0: individual runs
            'calc_directional': False,       # Calculate full BRDF
            'calc_vert_profiles': False,
            'soil_heat_method': 2,           # 2: G=0.35*Rn
            'calc_rss_rbs': False,           # 0: fixed resistance
            'MoninObukhov': True,            # Stability corrections
            'save_spectral': True,
            'calc_ebal': True,
        }

        print('SCOPE model initialized')
        print(f'  Input directory: {self.input_dir}')
        print(f'  Spectral bands: {len(self.spectral.wlS)}')

    def create_leafbio(self,
                       Cab: float = 40.0,
                       Cca: Optional[float] = None,
                       Cdm: float = 0.012,
                       Cw: float = 0.009,
                       Cs: float = 0.0,
                       Cant: float = 0.0,
                       Cp: float = 0.0,
                       Cbc: float = 0.0,
                       N: float = 1.4,
                       fqe: float = 0.01,
                       V2Z: float = 0,
                       Vcmax25: float = 60.0,
                       BallBerrySlope: float = 8.0,
                       BallBerry0: float = 0.01,
                       Type: str = 'C3',
                       RdPerVcmax25: float = 0.015,
                       Kn0: float = 2.48,
                       Knalpha: float = 2.83,
                       Knbeta: float = 0.114,
                       Tyear: float = 15.0,
                       beta: float = 0.51,
                       kNPQs: float = 0.0,
                       qLs: float = 1.0,
                       stressfactor: float = 1.0,
                       rho_thermal: float = 0.01,
                       tau_thermal: float = 0.01) -> LeafBio:
        """
        Create leaf biochemistry structure (SCOPE.m uses select_input).

        Default values match input_data_default.csv and numerical_experiment_matlab_simple.m
        """
        # Cca function of Cab if not provided (SCOPE.m line 135)
        if Cca is None:
            Cca = Cab / 4

        return LeafBio(
            Cab=Cab,
            Cca=Cca,
            Cdm=Cdm,
            Cw=Cw,
            Cs=Cs,
            Cant=Cant,
            Cp=Cp,
            Cbc=Cbc,
            N=N,
            fqe=fqe,
            V2Z=V2Z,
            Vcmax25=Vcmax25,
            BallBerrySlope=BallBerrySlope,
            BallBerry0=BallBerry0,
            Type=Type,
            RdPerVcmax25=RdPerVcmax25,
            Kn0=Kn0,
            Knalpha=Knalpha,
            Knbeta=Knbeta,
            Tyear=Tyear,
            beta=beta,
            kNPQs=kNPQs,
            qLs=qLs,
            stressfactor=stressfactor,
            rho_thermal=rho_thermal,
            tau_thermal=tau_thermal,
        )

    def create_canopy(self,
                      LAI: float = 3.0,
                      hc: float = 2.0,
                      LIDFa: float = -0.35,
                      LIDFb: float = -0.15,
                      leafwidth: float = 0.1,
                      rb: float = 10.0,
                      Cd: float = 0.3,
                      CR: float = 0.35,
                      CD1: float = 20.6,
                      Psicor: float = 0.2,
                      rwc: float = 1.0,
                      kV: float = 0.6396,
                      Rin: float = 600.0,
                      MoninObukhov: bool = False) -> Canopy:
        """
        Create canopy structure (SCOPE.m lines 265-270).

        Default values match input_data_default.csv and numerical_experiment_matlab_simple.m
        """
        # Calculate nlayers (SCOPE.m line 265)
        # nlayers = ceil(10*LAI) + ((Rin < 200) & MoninObukhov)*60
        nlayers = int(np.ceil(10 * LAI))
        if Rin < 200 and MoninObukhov:
            nlayers += 60
        nlayers = max(2, nlayers)  # patch for LAI < 0.1

        # Calculate derived parameters (SCOPE.m lines 160-163, 196-198)
        hot = leafwidth / hc
        zo = 0.25 * hc
        d = 0.65 * hc

        canopy = Canopy(
            LAI=LAI,
            hc=hc,
            LIDFa=LIDFa,
            LIDFb=LIDFb,
            leafwidth=leafwidth,
            rb=rb,
            Cd=Cd,
            CR=CR,
            CD1=CD1,
            Psicor=Psicor,
            rwc=rwc,
            zo=zo,
            d=d,
        )

        # Set internal attributes (SCOPE.m lines 267-269)
        canopy._nlayers = nlayers
        canopy._x = -np.arange(1, nlayers + 1) / nlayers  # column vector
        canopy._xl = np.concatenate([[0], canopy._x])     # add top level
        canopy._hot = hot
        canopy._kV = kV

        return canopy

    def create_soil(self,
                    spectrum: int = 1,
                    rs_thermal: float = 0.06,
                    SMC: float = 0.25,
                    rss: float = 500.0,
                    rbs: float = 10.0,
                    cs: float = 1180.0,
                    rhos: float = 1800.0,
                    CSSOIL: float = 0.01,
                    lambdas: float = 1.55,
                    BSMBrightness: float = 0.5,
                    BSMlat: float = 25.0,
                    BSMlon: float = 45.0,
                    refl_value: Optional[float] = None,
                    Ta: float = 20.0) -> Soil:
        """
        Create soil structure (SCOPE.m lines 341-346).

        Args:
            refl_value: If provided, use constant reflectance. Otherwise use spectrum file.
            Ta: Air temperature for initializing Tsold
        """
        soil = Soil(
            spectrum=spectrum,
            rs_thermal=rs_thermal,
            SMC=SMC,
            rss=rss,
            rbs=rbs,
            cs=cs,
            rhos=rhos,
            CSSOIL=CSSOIL,
            lambdas=lambdas,
            BSMBrightness=BSMBrightness,
            BSMlat=BSMlat,
            BSMlon=BSMlon,
        )

        # Initialize Tsold (SCOPE.m line 217)
        soil.Tsold = Ta * np.ones((12, 2))

        # Set soil reflectance
        nwl = len(self.spectral.wlS)
        if refl_value is not None:
            # Use constant reflectance
            soil_refl = refl_value * np.ones(nwl)
        else:
            # Default: load from file or use 0.1
            soil_refl = 0.1 * np.ones(nwl)

        # Set thermal reflectance (SCOPE.m line 346)
        soil_refl[self.spectral.IwlT] = rs_thermal
        soil.refl = soil_refl

        return soil

    def create_meteo(self,
                     z: float = 10.0,
                     Rin: float = 600.0,
                     Rli: float = 300.0,
                     Ta: float = 20.0,
                     p: float = 970.0,
                     ea: float = 15.0,
                     u: float = 2.0,
                     Ca: float = 410.0,
                     Oa: float = 209.0) -> Meteo:
        """
        Create meteorological structure.

        Default values match input_data_default.csv
        """
        return Meteo(
            z=z,
            Rin=Rin,
            Rli=Rli,
            Ta=Ta,
            p=p,
            ea=ea,
            u=u,
            Ca=Ca,
            Oa=Oa,
        )

    def create_angles(self,
                      tts: float = 30.0,
                      tto: float = 0.0,
                      psi: float = 0.0) -> Angles:
        """
        Create angles structure.

        Default values match input_data_default.csv
        """
        return Angles(tts=tts, tto=tto, psi=psi)

    def create_options(self, **kwargs) -> Options:
        """
        Create options structure.

        Keyword arguments override default_options.
        """
        opts = self.default_options.copy()
        opts.update(kwargs)

        return Options(
            calc_fluor=opts.get('calc_fluor', True),
            calc_planck=opts.get('calc_planck', True),
            calc_xanthophyll=opts.get('calc_xanthophyll', False),
            Fluorescence_model=opts.get('Fluorescence_model', 0),
            calc_directional=opts.get('calc_directional', False),
            calc_vert_profiles=opts.get('calc_vert_profiles', False),
            calc_ebal=opts.get('calc_ebal', True),
            lite=opts.get('lite', True),
            verify=opts.get('verify', False),
            apply_T_corr=opts.get('apply_T_corr', True),
            MoninObukhov=opts.get('MoninObukhov', True),
            soil_heat_method=opts.get('soil_heat_method', 2),
            calc_rss_rbs=opts.get('calc_rss_rbs', False),
        )

    def _aggreg_modtran(self, atmfile: Path) -> np.ndarray:
        """
        Aggregate MODTRAN data over SCOPE bands (like aggreg.m).

        Args:
            atmfile: Path to .atm file

        Returns:
            M: Aggregated matrix (nwl, 6) with transmission columns
        """
        # Read .atm file (skip header lines)
        with open(atmfile, 'r') as f:
            lines = f.readlines()

        # Find data start (skip header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith(('WN', '#', 'toasun')):
                try:
                    float(line.split()[0])
                    data_start = i
                    break
                except ValueError:
                    continue

        # Parse data
        data = []
        for line in lines[data_start:]:
            parts = line.split()
            if len(parts) >= 20:
                try:
                    values = [float(x) for x in parts[:20]]
                    data.append(values)
                except ValueError:
                    continue

        data = np.array(data)
        wlM = data[:, 1]  # Wavelength in nm
        T = data[:, 2:20]  # Transmission columns

        # Extract 6 relevant columns (matching MATLAB aggreg.m):
        # 1: t1 (toasun), 3: t3 (rdd), 4: t4 (tss), 5: t5 (tsd), 12: t12 (tssrdd), 16: t16 (Lab)
        U = np.column_stack([T[:, 0], T[:, 2], T[:, 3], T[:, 4], T[:, 11], T[:, 15]])

        # Get SCOPE spectral regions from SpectralBands
        nreg = self.spectral.nreg
        streg = np.array(self.spectral.start)
        enreg = np.array(self.spectral.end)
        width = np.array(self.spectral.res)

        # Number of bands in each region
        nwreg = ((enreg - streg) / width + 1).astype(int)

        # Offset for each region
        off = np.zeros(nreg, dtype=int)
        for i in range(1, nreg):
            off[i] = off[i-1] + nwreg[i-1]

        nwS = int(np.sum(nwreg))
        n = np.zeros(nwS)  # Count
        S = np.zeros((nwS, 6))  # Sums

        # Aggregate MODTRAN data to SCOPE bands
        for iwl in range(len(wlM)):
            w = wlM[iwl]
            for r in range(nreg):
                j = int(round((w - streg[r]) / width[r]))
                if 0 <= j < nwreg[r]:
                    k = j + off[r]
                    S[k, :] += U[iwl, :]
                    n[k] += 1

        # Calculate averages
        M = np.zeros((nwS, 6))
        for i in range(6):
            with np.errstate(divide='ignore', invalid='ignore'):
                M[:, i] = np.where(n > 0, S[:, i] / n, 0)

        return M

    def load_atmo(self, atmfile: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load atmospheric irradiance from file (like SCOPE.m load_atmo.m).

        This is the default method matching SCOPE.m behavior.

        Args:
            atmfile: Path to atmospheric file. If None, loads default
                     FLEX-S3_std.atm from input/radiationdata/
                     Supports .atm (MODTRAN) and .dat (direct Esun/Esky) formats.

        Returns:
            Dictionary with either:
            - 'M': MODTRAN matrix for calcTOCirr processing (scaled by Rin)
            - 'Esun_', 'Esky_': Direct spectral irradiance arrays
        """
        rad_dir = self.input_dir / 'radiationdata'

        if atmfile is None:
            # Default: try MODTRAN file first (SCOPE.m default), then fall back to .dat
            modtran_file = rad_dir / 'FLEX-S3_std.atm'
            if modtran_file.exists():
                atmfile = modtran_file
            else:
                esun_file = rad_dir / 'Esun_.dat'
                esky_file = rad_dir / 'Esky_.dat'
                if esun_file.exists() and esky_file.exists():
                    Esun_ = np.loadtxt(esun_file)
                    Esky_ = np.loadtxt(esky_file)
                    return {'Esun_': Esun_, 'Esky_': Esky_}
                else:
                    raise FileNotFoundError(
                        f"No atmospheric files found in {rad_dir}\n"
                        "Use create_atmo_simple() for simplified model."
                    )
        else:
            atmfile = Path(atmfile)
            if not atmfile.exists():
                atmfile = rad_dir / atmfile
            if not atmfile.exists():
                raise FileNotFoundError(f"Atmospheric file not found: {atmfile}")

        # Check file extension
        if atmfile.suffix == '.atm':
            # MODTRAN format - aggregate and return M matrix
            M = self._aggreg_modtran(atmfile)
            return {'M': M}
        else:
            # Direct format (2 columns: Esun, Esky)
            data = np.loadtxt(atmfile)
            if data.ndim == 1:
                raise ValueError("Atmospheric file must have 2 columns (Esun, Esky)")
            Esun_ = data[:, 0]
            Esky_ = data[:, 1]

            nwl = len(self.spectral.wlS)
            if len(Esun_) != nwl:
                raise ValueError(
                    f"Atmospheric data length ({len(Esun_)}) doesn't match "
                    f"spectral grid ({nwl})"
                )
            return {'Esun_': Esun_, 'Esky_': Esky_}

    def create_atmo_simple(self, Rin: float, Rli: float = 300.0) -> Dict[str, np.ndarray]:
        """
        Create simplified atmospheric irradiance (for numerical experiments).

        Uses 70/30 direct/diffuse split with Gaussian spectral shape.
        This matches numerical_experiment_matlab_simple.m's create_simple_atmo().

        Note: For full SCOPE.m compatibility, use load_atmo() instead.

        Args:
            Rin: Incoming shortwave radiation [W/m²]
            Rli: Incoming longwave radiation [W/m²]

        Returns:
            Dictionary with Esun_ and Esky_ arrays
        """
        nwl = len(self.spectral.wlS)
        wl = self.spectral.wlS

        Esun_ = np.zeros(nwl)
        Esky_ = np.zeros(nwl)

        # Optical region (wl < 3000 nm)
        iwl = wl < 3000
        if np.any(iwl) and Rin > 0:
            wl_opt = wl[iwl]
            spec = np.exp(-0.5 * ((wl_opt - 500) / 300) ** 2)
            tot = np.trapz(spec, wl_opt)
            if tot > 0:
                Esun_[iwl] = 0.7 * Rin * 1000 / tot * spec  # 70% direct
                Esky_[iwl] = 0.3 * Rin * 1000 / tot * spec  # 30% diffuse

        # Thermal region (wl >= 3000 nm)
        iwl = wl >= 3000
        if np.any(iwl) and Rli > 0:
            wl_therm = wl[iwl]
            spec = np.exp(-0.5 * ((wl_therm - 10000) / 5000) ** 2)
            tot = np.trapz(spec, wl_therm)
            if tot > 0:
                Esky_[iwl] = Rli * 1000 / tot * spec
                Esun_[iwl] = 1e-10

        return {'Esun_': Esun_, 'Esky_': Esky_}

    def create_atmo(self, Rin: float = None, Rli: float = 300.0,
                    use_file: bool = True) -> Dict[str, np.ndarray]:
        """
        Create atmospheric irradiance structure.

        By default, loads from files like SCOPE.m does. Falls back to
        simplified model if files not available.

        Args:
            Rin: Incoming shortwave radiation [W/m²] (only used if use_file=False)
            Rli: Incoming longwave radiation [W/m²] (only used if use_file=False)
            use_file: If True, load from input/radiationdata/ (SCOPE.m default)
                      If False, use simplified Gaussian model

        Returns:
            Dictionary with Esun_ and Esky_ arrays
        """
        if use_file:
            try:
                return self.load_atmo()
            except FileNotFoundError:
                print("Warning: Atmospheric files not found, using simplified model")
                if Rin is None:
                    Rin = 600.0  # Default
                return self.create_atmo_simple(Rin, Rli)
        else:
            if Rin is None:
                raise ValueError("Rin is required when use_file=False")
            return self.create_atmo_simple(Rin, Rli)

    def run(self,
            leafbio: LeafBio,
            canopy: Canopy,
            soil: Soil,
            meteo: Meteo,
            angles: Angles,
            options: Optional[Options] = None,
            atmo: Optional[Dict] = None,
            xyt: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run SCOPE model (main workflow from SCOPE.m lines 284-459).

        Args:
            leafbio: Leaf biochemistry structure
            canopy: Canopy structure
            soil: Soil structure
            meteo: Meteorological structure
            angles: Angles structure
            options: Model options (default: create from default_options)
            atmo: Atmospheric irradiance (default: create from meteo.Rin, meteo.Rli)
            xyt: Time series info (default: None for single run)

        Returns:
            Dictionary with all model outputs
        """
        # Default options
        if options is None:
            options = self.create_options()

        # Integration method (SCOPE.m lines 77-80)
        integr = 'layers' if options.lite else 'angles_and_layers'

        # Number of layers
        nl = canopy.nlayers

        # Create atmosphere if not provided
        if atmo is None:
            atmo = self.create_atmo(meteo.Rin, meteo.Rli)

        # XYT structure for time series (SCOPE.m line 229)
        if xyt is None:
            xyt = {'t': 0, 'year': 2024}

        # 1. Compute leaf angle distribution (SCOPE.m line 286)
        compute_canopy_lidf(canopy)

        # 2. Leaf radiative transfer model FLUSPECT (SCOPE.m lines 289-338)
        leafbio_emis = 1 - leafbio.rho_thermal - leafbio.tau_thermal

        # Run Fluspect (SCOPE.m line 327)
        leafopt = fluspect(
            leafbio=leafbio,
            optipar=self.optipar,
            wlP=self.spectral.wlP,
            wlE=self.spectral.wlE if options.calc_fluor else None,
            wlF=self.spectral.wlF if options.calc_fluor else None,
        )

        # Extend to full spectrum if needed
        if len(leafopt.refl) == len(self.spectral.wlP):
            refl_full = np.interp(self.spectral.wlS, self.spectral.wlP, leafopt.refl, left=0.05, right=0.05)
            tran_full = np.interp(self.spectral.wlS, self.spectral.wlP, leafopt.tran, left=0.05, right=0.01)
            leafopt.refl = refl_full
            leafopt.tran = tran_full

        # Set thermal properties (SCOPE.m lines 328-329)
        leafopt.refl[self.spectral.IwlT] = leafbio.rho_thermal
        leafopt.tran[self.spectral.IwlT] = leafbio.tau_thermal

        # 3. Four stream canopy radiative transfer model RTMo (SCOPE.m line 349)
        options_dict = asdict(options)
        rad, gap, profiles = rtmo(
            spectral=self.spectral,
            atmo=atmo,
            soil=soil,
            leafopt=leafopt,
            canopy=canopy,
            angles=angles,
            meteo=meteo,
            options=options_dict,
        )

        # 4. Energy balance (SCOPE.m lines 352-354)
        k = 1  # iteration counter
        iter_count, rad, thermal, soil, bcu, bch, fluxes, resistance, meteo = ebal(
            constants=self.constants,
            options=options,
            rad=rad,
            gap=gap,
            meteo=meteo,
            soil=soil,
            canopy=canopy,
            leafbio=leafbio,
            k=k,
            xyt=xyt,
            integr=integr,
        )

        # 5. Fluorescence radiative transfer model RTMf (SCOPE.m lines 357-359)
        fluor_output = None
        if options.calc_fluor and leafopt.Mb is not None:
            # Get eta from energy balance (SCOPE.m line 358)
            etau = bcu.eta if hasattr(bcu, 'eta') and bcu.eta is not None else np.ones(nl)
            etah = bch.eta if hasattr(bch, 'eta') and bch.eta is not None else np.ones(nl)

            fluor_output = rtmf(
                spectral=self.spectral,
                rad=rad,
                soil=soil,
                leafopt=leafopt,
                canopy=canopy,
                gap=gap,
                angles=angles,
                etau=etau,
                etah=etah,
            )

        # 5b. Escape ratio (EoutF_ / Femleaves_)
        if fluor_output is not None and hasattr(fluor_output, 'EoutF_') and hasattr(fluor_output, 'Femleaves_'):
            with np.errstate(divide='ignore', invalid='ignore'):
                escape_ratio_ = np.where(fluor_output.Femleaves_ > 0,
                                         fluor_output.EoutF_ / fluor_output.Femleaves_, np.nan)
            # wlF starts at 640nm, 1nm steps (0-based): 685nm=45, 740nm=100, 761nm=121
            escape_685 = escape_ratio_[45] if len(escape_ratio_) > 45 else np.nan
            escape_740 = escape_ratio_[100] if len(escape_ratio_) > 100 else np.nan
            escape_761 = escape_ratio_[121] if len(escape_ratio_) > 121 else np.nan
        else:
            escape_685 = np.nan
            escape_740 = np.nan
            escape_761 = np.nan

        # 6. Computation of data products (SCOPE.m lines 377-442)
        Ps = gap.Ps[:nl]
        Ph = 1 - Ps

        # LAI sunlit/shaded (SCOPE.m lines 384-385)
        canopy_LAIsunlit = canopy.LAI * np.mean(Ps)
        canopy_LAIshaded = canopy.LAI - canopy_LAIsunlit

        # Photosynthesis (SCOPE.m lines 416-417)
        if hasattr(bcu, 'A') and hasattr(bch, 'A'):
            A_mean = np.mean(bch.A * Ph) + np.mean(bcu.A * Ps)
            canopy_A = canopy.LAI * A_mean
        else:
            canopy_A = np.nan

        if hasattr(bcu, 'Ag') and hasattr(bch, 'Ag'):
            Ag_mean = np.mean(bch.Ag * Ph) + np.mean(bcu.Ag * Ps)
            canopy_GPP = canopy.LAI * Ag_mean
        else:
            canopy_GPP = np.nan

        # Build output dictionary
        output = {
            # Core outputs
            'rad': rad,
            'gap': gap,
            'profiles': profiles,
            'thermal': thermal,
            'fluxes': fluxes,
            'resistance': resistance,
            'bcu': bcu,
            'bch': bch,
            'leafopt': leafopt,

            # Fluorescence outputs
            'fluorescence': fluor_output,

            # Derived canopy products
            'canopy': {
                'LAIsunlit': canopy_LAIsunlit,
                'LAIshaded': canopy_LAIshaded,
                'A': canopy_A,
                'GPP': canopy_GPP,
            },

            # Iteration info
            'iter': iter_count,

            # Extracted key values for convenience
            'F684': getattr(fluor_output, 'F684', np.nan) if fluor_output else np.nan,
            'F685': fluor_output.F685 if fluor_output else np.nan,
            'F740': fluor_output.F740 if fluor_output else np.nan,
            'F761': getattr(fluor_output, 'F761', np.nan) if fluor_output else np.nan,
            'wl685': getattr(fluor_output, 'wl685', np.nan) if fluor_output else np.nan,
            'wl740': getattr(fluor_output, 'wl740', np.nan) if fluor_output else np.nan,
            'escape_685': escape_685,
            'escape_740': escape_740,
            'escape_761': escape_761,
            'LoutF': fluor_output.LoutF if fluor_output else np.nan,
            'EoutF': fluor_output.EoutF if fluor_output else np.nan,
            'Eouto': getattr(rad, 'Eouto', np.nan),
            'Rntot': fluxes.get('Rntot', np.nan),
            'lEtot': fluxes.get('lEtot', np.nan),
            'Htot': fluxes.get('Htot', np.nan),
            'Actot': fluxes.get('Actot', np.nan),
            'Rnctot': fluxes.get('Rnctot', np.nan),
            'Tcave': fluxes.get('Tcave', np.nan),
            'Tsave': fluxes.get('Tsave', np.nan),
        }

        return output

    def verify_output(
        self,
        verification_dir: Optional[str] = None,
        n_simulations: Optional[int] = None,
        tolerance: float = 1e-9,
        rel_tolerance: Optional[float] = None,
        verbose: bool = True,
    ) -> bool:
        """
        Verify model outputs against verification dataset (like output_verification_csv.m).

        This method runs the model with verification input parameters and compares
        the outputs against stored verification data. It prints "The output is the
        same as in the verification data set" if all comparisons pass.

        Args:
            verification_dir: Path to verification data directory
                              (default: output/verificationdata)
            n_simulations: Number of simulations to verify (default: all)
            tolerance: Threshold for sum of squared differences (default: 1e-9)
                       Used when rel_tolerance is None (exact match mode)
            rel_tolerance: Relative tolerance for comparison (e.g., 0.01 = 1%)
                          If provided, uses relative difference instead of
                          sum of squared differences
            verbose: Print progress and detailed results

        Returns:
            True if verification passes, False otherwise
        """
        # Default verification directory
        if verification_dir is None:
            verification_dir = self.scope_dir / 'output' / 'verificationdata'
        else:
            verification_dir = Path(verification_dir)

        if not verification_dir.exists():
            print(f"Warning: Verification directory not found: {verification_dir}")
            return False

        if verbose:
            print(f"Loading verification data from: {verification_dir}")

        # Load verification input parameters (pars_and_input_short.csv)
        pars_file = verification_dir / 'pars_and_input_short.csv'
        if not pars_file.exists():
            print(f"Warning: Input parameters file not found: {pars_file}")
            return False

        # Read CSV, skip units row (row 1)
        input_df = pd.read_csv(pars_file, skiprows=[1])

        # Load verification outputs
        fluxes_file = verification_dir / 'fluxes.csv'
        fluor_file = verification_dir / 'fluorescence_scalars.csv'

        verification_outputs = {}
        if fluxes_file.exists():
            verification_outputs['fluxes'] = pd.read_csv(fluxes_file, skiprows=[1])
        if fluor_file.exists():
            verification_outputs['fluorescence'] = pd.read_csv(fluor_file, skiprows=[1])

        if not verification_outputs:
            print("Warning: No verification output files found")
            return False

        # Determine number of simulations
        n_total = len(input_df)
        if n_simulations is None:
            n_simulations = n_total
        else:
            n_simulations = min(n_simulations, n_total)

        if verbose:
            print(f"Running {n_simulations} verification simulations...")

        # Track differences
        different_content = False
        different_results: List[Dict] = []

        # Run simulations and compare
        for i in range(n_simulations):
            row = input_df.iloc[i]

            if verbose and (i + 1) % 10 == 0:
                print(f"  Simulation {i + 1}/{n_simulations}...")

            try:
                # Create input structures from verification parameters
                leafbio = self.create_leafbio(
                    Cab=row.get('Cab', 40.0),
                    Cca=row.get('Cca', 10.0),
                    Cdm=row.get('Cdm', 0.012),
                    Cw=row.get('Cw', 0.009),
                    Cs=row.get('Cs', 0.0),
                    Cant=row.get('Cant', 0.0),
                    N=row.get('N', 1.4),
                    Vcmax25=row.get('Vcmax25', 60.0),
                    BallBerrySlope=row.get('BallBerrySlope', 8.0),
                )

                canopy = self.create_canopy(
                    LAI=row.get('LAI', 3.0),
                    LIDFa=row.get('LIDFa', -0.35),
                    LIDFb=row.get('LIDFb', 0.0),
                    Rin=row.get('Rin', 600.0),
                )

                soil = self.create_soil(Ta=row.get('Ta', 20.0))

                meteo = self.create_meteo(
                    Rin=row.get('Rin', 600.0),
                    Ta=row.get('Ta', 20.0),
                    ea=row.get('ea', 15.0),
                )

                angles = self.create_angles(
                    tts=row.get('tts', 30.0),
                    tto=row.get('tto', 0.0),
                    psi=row.get('psi', 0.0),
                )

                options = self.create_options(lite=True, calc_fluor=True, calc_ebal=True)

                # Run model
                output = self.run(
                    leafbio=leafbio,
                    canopy=canopy,
                    soil=soil,
                    meteo=meteo,
                    angles=angles,
                    options=options,
                )

                # Compare outputs with verification data
                # Check fluxes
                if 'fluxes' in verification_outputs and i < len(verification_outputs['fluxes']):
                    v_fluxes = verification_outputs['fluxes'].iloc[i]

                    comparisons = [
                        ('Rntot', output.get('Rntot', np.nan), v_fluxes.get('Rntot', np.nan)),
                        ('lEtot', output.get('lEtot', np.nan), v_fluxes.get('lEtot', np.nan)),
                        ('Htot', output.get('Htot', np.nan), v_fluxes.get('Htot', np.nan)),
                        ('Actot', output.get('Actot', np.nan), v_fluxes.get('Actot', np.nan)),
                        ('Tcave', output.get('Tcave', np.nan), v_fluxes.get('Tcave', np.nan)),
                        ('Tsave', output.get('Tsave', np.nan), v_fluxes.get('Tsave', np.nan)),
                    ]

                    for var_name, py_val, mat_val in comparisons:
                        if np.isnan(py_val) and np.isnan(mat_val):
                            continue
                        if np.isnan(py_val) or np.isnan(mat_val):
                            # One is NaN, one is not - treat as different
                            different_content = True
                            different_results.append({
                                'simulation': i + 1,
                                'variable': var_name,
                                'python': py_val,
                                'matlab': mat_val,
                                'diff': np.nan,
                            })
                            continue

                        # Calculate difference
                        if rel_tolerance is not None:
                            # Use relative difference
                            if abs(mat_val) > 1e-10:
                                diff = abs(py_val - mat_val) / abs(mat_val)
                            else:
                                diff = abs(py_val - mat_val)
                            is_different = diff > rel_tolerance
                        else:
                            # Use sum of squared differences (MATLAB default)
                            diff = (py_val - mat_val) ** 2
                            is_different = diff > tolerance

                        if is_different:
                            different_content = True
                            different_results.append({
                                'simulation': i + 1,
                                'variable': var_name,
                                'python': py_val,
                                'matlab': mat_val,
                                'diff': diff,
                            })

                # Check fluorescence
                if 'fluorescence' in verification_outputs and i < len(verification_outputs['fluorescence']):
                    v_fluor = verification_outputs['fluorescence'].iloc[i]

                    fluor_comparisons = [
                        ('F684', output.get('F685', np.nan), v_fluor.get('F684', np.nan)),  # F685 vs F684
                        ('F761', output.get('F761', np.nan), v_fluor.get('F761', np.nan)),
                    ]

                    for var_name, py_val, mat_val in fluor_comparisons:
                        if np.isnan(py_val) and np.isnan(mat_val):
                            continue
                        if np.isnan(py_val) or np.isnan(mat_val):
                            different_content = True
                            different_results.append({
                                'simulation': i + 1,
                                'variable': var_name,
                                'python': py_val,
                                'matlab': mat_val,
                                'diff': np.nan,
                            })
                            continue

                        if rel_tolerance is not None:
                            if abs(mat_val) > 1e-10:
                                diff = abs(py_val - mat_val) / abs(mat_val)
                            else:
                                diff = abs(py_val - mat_val)
                            is_different = diff > rel_tolerance
                        else:
                            diff = (py_val - mat_val) ** 2
                            is_different = diff > tolerance

                        if is_different:
                            different_content = True
                            different_results.append({
                                'simulation': i + 1,
                                'variable': var_name,
                                'python': py_val,
                                'matlab': mat_val,
                                'diff': diff,
                            })

            except Exception as e:
                if verbose:
                    print(f"  Simulation {i + 1} failed: {e}")
                different_content = True

        # Print results (following MATLAB output_verification_csv.m format)
        if different_content:
            print("\nWarning: data in the output are different from the verification output")
            if verbose and different_results:
                diff_label = "rel_diff" if rel_tolerance is not None else "sq_diff"
                print(f"\nDifferences found (tolerance: {rel_tolerance if rel_tolerance else tolerance}):")
                for r in different_results[:10]:  # Show first 10
                    print(f"  Sim {r['simulation']}, {r['variable']}: "
                          f"Python={r['python']:.6e}, MATLAB={r['matlab']:.6e}, "
                          f"{diff_label}={r['diff']:.2e}")
                if len(different_results) > 10:
                    print(f"  ... and {len(different_results) - 10} more differences")
            return False
        else:
            print("The output is the same as in the verification data set")
            return True


def run_scope(
    # Leaf parameters
    Cab: float = 40.0,
    Cca: Optional[float] = None,
    Cdm: float = 0.012,
    Cw: float = 0.009,
    Cs: float = 0.0,
    N: float = 1.4,
    fqe: float = 0.01,
    Vcmax25: float = 60.0,
    # Canopy parameters
    LAI: float = 3.0,
    hc: float = 2.0,
    LIDFa: float = -0.35,
    LIDFb: float = -0.15,
    # Angles
    tts: float = 30.0,
    tto: float = 0.0,
    psi: float = 0.0,
    # Meteo
    Rin: float = 600.0,
    Rli: float = 300.0,
    Ta: float = 20.0,
    p: float = 970.0,
    ea: float = 15.0,
    u: float = 2.0,
    Ca: float = 410.0,
    # Soil
    soil_refl: float = 0.1,
    # Options
    lite: bool = True,
    calc_fluor: bool = True,
    calc_ebal: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run SCOPE with common parameters.

    This provides a simple interface similar to calling SCOPE.m with default settings.

    Returns:
        Dictionary with model outputs including F740, F685, Eouto, etc.
    """
    # Initialize model
    model = SCOPEModel()

    # Create input structures
    leafbio = model.create_leafbio(
        Cab=Cab, Cca=Cca, Cdm=Cdm, Cw=Cw, Cs=Cs, N=N,
        fqe=fqe, Vcmax25=Vcmax25,
    )

    canopy = model.create_canopy(
        LAI=LAI, hc=hc, LIDFa=LIDFa, LIDFb=LIDFb, Rin=Rin,
    )

    soil = model.create_soil(refl_value=soil_refl, Ta=Ta)

    meteo = model.create_meteo(
        Rin=Rin, Rli=Rli, Ta=Ta, p=p, ea=ea, u=u, Ca=Ca,
    )

    angles = model.create_angles(tts=tts, tto=tto, psi=psi)

    options = model.create_options(
        lite=lite, calc_fluor=calc_fluor, calc_ebal=calc_ebal,
    )

    # Run model
    output = model.run(
        leafbio=leafbio,
        canopy=canopy,
        soil=soil,
        meteo=meteo,
        angles=angles,
        options=options,
    )

    return output


# Main entry point for command-line usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run SCOPE model')
    parser.add_argument('--Cab', type=float, default=40.0, help='Chlorophyll content [μg/cm²]')
    parser.add_argument('--LAI', type=float, default=3.0, help='Leaf Area Index')
    parser.add_argument('--tts', type=float, default=30.0, help='Solar zenith angle [deg]')
    parser.add_argument('--tto', type=float, default=0.0, help='View zenith angle [deg]')
    parser.add_argument('--Rin', type=float, default=600.0, help='Incoming radiation [W/m²]')
    parser.add_argument('--Ta', type=float, default=20.0, help='Air temperature [°C]')
    parser.add_argument('--verify', action='store_true',
                        help='Run verification against reference dataset')
    parser.add_argument('--verify-n', type=int, default=None,
                        help='Number of verification simulations (default: all)')
    parser.add_argument('--verify-tol', type=float, default=None,
                        help='Relative tolerance for verification (e.g., 0.05 = 5%%). '
                             'If not set, uses strict sum-of-squares tolerance.')

    args = parser.parse_args()

    print('=' * 60)
    print('SCOPE Python Model')
    print('=' * 60)

    # Initialize model
    model = SCOPEModel()

    if args.verify:
        # Run verification mode
        print('\n=== Verification Mode ===')
        success = model.verify_output(
            n_simulations=args.verify_n,
            rel_tolerance=args.verify_tol,
            verbose=True,
        )
        print('=' * 60)
        sys.exit(0 if success else 1)
    else:
        # Run model with specified parameters (reuse the model instance)
        leafbio = model.create_leafbio(Cab=args.Cab)
        canopy = model.create_canopy(LAI=args.LAI, Rin=args.Rin)
        soil = model.create_soil(Ta=args.Ta)
        meteo = model.create_meteo(Rin=args.Rin, Ta=args.Ta)
        angles = model.create_angles(tts=args.tts, tto=args.tto)
        options = model.create_options(lite=True, calc_fluor=True, calc_ebal=True)

        output = model.run(
            leafbio=leafbio,
            canopy=canopy,
            soil=soil,
            meteo=meteo,
            angles=angles,
            options=options,
        )

        # Print results
        print('\n=== Results ===')
        print(f'F685: {output["F685"]:.4f} mW/m²/nm')
        print(f'F740: {output["F740"]:.4f} mW/m²/nm')
        print(f'F761: {output["F761"]:.4f} mW/m²/nm')
        print(f'Eouto: {output["Eouto"]:.4f} W/m²')
        print(f'Rntot: {output["Rntot"]:.4f} W/m²')
        print(f'lEtot: {output["lEtot"]:.4f} W/m²')
        print(f'Htot: {output["Htot"]:.4f} W/m²')
        print(f'Actot: {output["Actot"]:.4f} μmol/m²/s')
        print(f'Tcave: {output["Tcave"]:.2f} °C')
        print(f'Tsave: {output["Tsave"]:.2f} °C')
        print('=' * 60)
