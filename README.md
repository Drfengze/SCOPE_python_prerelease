# SCOPE Python

Python implementation of the [SCOPE](https://scope-model.readthedocs.io) (Soil Canopy Observation, Photochemistry and Energy fluxes) model.

## Overview

SCOPE is a vertical (1-D) integrated radiative transfer and energy balance model for vegetated land surfaces. It simulates:

- **Leaf and canopy optical properties** (PROSPECT + SAIL)
- **Solar-induced fluorescence (SIF)** emission
- **Photosynthesis** (Farquhar model) and stomatal conductance (Ball-Berry)
- **Energy balance** and surface temperature
- **Outgoing radiation**: reflected, emitted, and fluorescence
- **Directional reflectance** (BRDF) and vegetation indices (PRI)

## Installation

```bash
pip install scope-model
```

For development:

```bash
git clone https://github.com/Drfengze/SCOPE_python_prerelease.git
cd SIF/SCOPE-master/python
pip install -e ".[dev]"
```

## Quick Start

```python
import scope
from scope.main import run_scope
from scope.types import Angles, Canopy, LeafBio, Meteo, Options, Soil
leafbio = LeafBio(Cab=40.0, Cca=10.0, Vcmax25=60.0)
canopy = Canopy(LAI=3.0, hc=2.0)
soil = Soil()
meteo = Meteo(Rin=600.0, Rli=300.0, Ta=20.0, p=970.0, ea=15.0, u=2.0, Ca=410.0)
angles = Angles(tts=30.0, tto=0.0, psi=0.0)
options = Options(lite=True, calc_fluor=True, calc_ebal=True)
output = run_scope(
    leafbio=leafbio,
    canopy=canopy,
    soil=soil,
    meteo=meteo,
    angles=angles,
    options=options,
)
summary = {
    "F685": getattr(output.fluorescence, "F685", None),
    "F740": getattr(output.fluorescence, "F740", None),
    "F761": getattr(output.fluorescence, "F761", None),
    "Eouto": getattr(output.rad, "Eouto", None),
    "Rntot": output.fluxes.get("Rntot"),
    "lEtot": output.fluxes.get("lEtot"),
    "Htot": output.fluxes.get("Htot"),
    "Actot": output.fluxes.get("Actot"),
    "Tcave": output.fluxes.get("Tcave"),
    "Tsave": output.fluxes.get("Tsave"),
}

summary
```

## Package Structure

```
scope/
├── constants.py      # Physical constants
├── spectral.py       # Spectral band definitions
├── main.py           # Main simulation runner
├── types/            # Typed dataclass structures
│   ├── angles.py     #   Solar/viewing geometry
│   ├── canopy.py     #   Canopy properties
│   ├── leafbio.py    #   Leaf biochemistry
│   ├── meteo.py      #   Meteorological data
│   ├── options.py    #   Simulation options
│   ├── radiation.py  #   Radiation outputs
│   └── soil.py       #   Soil properties
├── rtm/              # Radiative transfer models
│   ├── fluspect.py   #   Leaf optics (PROSPECT + fluorescence)
│   ├── bsm.py        #   Soil reflectance (BSM)
│   ├── rtmo.py       #   Canopy optical RT (SAIL-based)
│   ├── rtmf.py       #   Fluorescence RT
│   ├── rtmt.py       #   Thermal RT
│   └── rtmz.py       #   Xanthophyll/PRI
├── fluxes/           # Energy balance & biochemistry
│   ├── biochemical.py  # Farquhar photosynthesis
│   ├── ebal.py         # Energy balance iteration
│   ├── heatfluxes.py   # Sensible/latent heat
│   └── resistances.py  # Aerodynamic resistances
├── supporting/       # Utility functions
│   ├── physics.py    #   Planck, vapor pressure, etc.
│   ├── leafangles.py #   Leaf angle distributions
│   ├── meanleaf.py   #   Leaf averaging
│   └── integration.py #  Numerical integration
└── io/               # Input/output
    ├── config_loader.py   # Configuration loading
    ├── load_timeseries.py # Time series data
    ├── load_atmo.py       # Atmospheric data
    ├── load_optipar.py    # Optical parameters
    └── output_writer.py   # Output files
```

## Requirements

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- Pandas >= 2.0

## References

- van der Tol, C., Verhoef, W., Timmermans, J., Verhoef, A., & Su, Z. (2009).
  An integrated model of soil-canopy spectral radiances, photosynthesis,
  fluorescence, temperature and energy balance. *Biogeosciences*, 6(12), 3109-3129.
- Yang, P., et al. (2021). SCOPE 2.0: a model to simulate vegetated land surface
  fluxes and satellite signals. *Geoscientific Model Development*, 14, 4697-4712.

## License

GPL-3.0
