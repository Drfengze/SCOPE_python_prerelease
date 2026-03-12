"""Load optical parameters for Fluspect and BSM models.

Loads OptiPar data from MATLAB .mat files and soil spectra from text files.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from ..rtm.fluspect import OptiPar
from ..rtm.bsm import BSMSpectra


def load_optipar(
    filepath: Optional[str] = None,
    input_dir: Optional[str] = None,
) -> Tuple[OptiPar, NDArray[np.float64]]:
    """Load optical parameters from MATLAB .mat file.

    Args:
        filepath: Full path to OptiPar .mat file (optional)
        input_dir: SCOPE input directory containing fluspect_parameters/

    Returns:
        Tuple of (OptiPar, wlP wavelengths)
    """
    if filepath is None:
        if input_dir is None:
            # Default to SCOPE-master/input
            input_dir = Path(__file__).parent.parent.parent / "input"
        else:
            input_dir = Path(input_dir)

        filepath = input_dir / "fluspect_parameters" / "Optipar2021_ProspectPRO_CX.mat"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"OptiPar file not found: {filepath}")

    # Load MATLAB file
    mat = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
    op = mat['optipar']

    # Extract wavelengths
    wlP = np.asarray(op.wl).flatten()

    # Extract absorption coefficients
    optipar = OptiPar(
        nr=np.asarray(op.nr).flatten(),
        Kab=np.asarray(op.Kab).flatten(),
        Kca=np.asarray(op.Kca).flatten(),
        KcaV=np.asarray(op.KcaV).flatten() if hasattr(op, 'KcaV') else None,
        KcaZ=np.asarray(op.KcaZ).flatten() if hasattr(op, 'KcaZ') else None,
        Kw=np.asarray(op.Kw).flatten(),
        Kdm=np.asarray(op.Kdm).flatten(),
        Ks=np.asarray(op.Ks).flatten(),
        Kant=np.asarray(op.Kant).flatten() if hasattr(op, 'Kant') else None,
        Kp=np.asarray(op.Kp).flatten() if hasattr(op, 'Kp') else None,
        Kcbc=np.asarray(op.Kcbc).flatten() if hasattr(op, 'Kcbc') else None,
        phi=np.asarray(op.phi).flatten() if hasattr(op, 'phi') else None,
    )

    return optipar, wlP


def load_bsm_spectra(
    optipar_path: Optional[str] = None,
    input_dir: Optional[str] = None,
) -> BSMSpectra:
    """Load BSM spectral data from OptiPar file.

    The GSV, nw, and Kw data are stored in the same OptiPar .mat file.

    Args:
        optipar_path: Full path to OptiPar .mat file (optional)
        input_dir: SCOPE input directory

    Returns:
        BSMSpectra with GSV, Kw, nw
    """
    if optipar_path is None:
        if input_dir is None:
            input_dir = Path(__file__).parent.parent.parent / "input"
        else:
            input_dir = Path(input_dir)

        optipar_path = input_dir / "fluspect_parameters" / "Optipar2021_ProspectPRO_CX.mat"

    optipar_path = Path(optipar_path)
    if not optipar_path.exists():
        raise FileNotFoundError(f"OptiPar file not found: {optipar_path}")

    mat = loadmat(str(optipar_path), squeeze_me=True, struct_as_record=False)
    op = mat['optipar']

    # GSV is shape (nwl, 3)
    GSV = np.asarray(op.GSV)
    if GSV.ndim == 1:
        # If squeezed to 1D, need to reshape
        GSV = GSV.reshape(-1, 3)

    return BSMSpectra(
        GSV=GSV,
        Kw=np.asarray(op.Kw).flatten(),
        nw=np.asarray(op.nw).flatten(),
    )


def load_soil_spectra(
    filepath: Optional[str] = None,
    input_dir: Optional[str] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load soil reflectance spectra from text file.

    Args:
        filepath: Full path to soil spectra file (optional)
        input_dir: SCOPE input directory containing soil_spectra/

    Returns:
        Tuple of (wavelengths, soil_reflectance)
    """
    if filepath is None:
        if input_dir is None:
            input_dir = Path(__file__).parent.parent.parent / "input"
        else:
            input_dir = Path(input_dir)

        filepath = input_dir / "soil_spectra" / "soilnew.txt"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Soil spectra file not found: {filepath}")

    # Load soil data - assume format: wavelength, reflectance columns
    data = np.loadtxt(filepath)

    if data.ndim == 1:
        # Single column = just reflectance, assume standard wavelengths
        soil_refl = data
        wl = np.arange(400, 400 + len(data))
    else:
        # Two columns: wavelength, reflectance
        wl = data[:, 0]
        soil_refl = data[:, 1]

    return wl, soil_refl
