"""Fluspect leaf optical properties model.

Translated from: src/RTMs/fluspect_B_CX.m

Fluspect extends the PROSPECT leaf model with chlorophyll fluorescence.
It calculates leaf reflectance, transmittance, and fluorescence matrices.

References:
    Vilfan, N., van der Tol, C., Muller, O., Rascher, U., & Verhoef, W. (2016).
    Fluspect-B: A model for leaf fluorescence, reflectance and transmittance
    spectra. Remote Sensing of Environment, 186, 596-615.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import expi as expint


@dataclass
class OptiPar:
    """Optical parameters for Fluspect model.

    These are the specific absorption coefficients for various leaf
    constituents at each wavelength.

    Attributes:
        nr: Refractive index spectrum, shape (nwl,)
        Kab: Chlorophyll a+b absorption, shape (nwl,)
        Kca: Carotenoid absorption, shape (nwl,)
        KcaV: Violaxanthin absorption, shape (nwl,)
        KcaZ: Zeaxanthin absorption, shape (nwl,)
        Kw: Water absorption, shape (nwl,)
        Kdm: Dry matter absorption, shape (nwl,)
        Ks: Senescent material absorption, shape (nwl,)
        Kant: Anthocyanin absorption, shape (nwl,)
        Kp: Protein absorption, shape (nwl,) (PROSPECT-PRO)
        Kcbc: Carbon-based constituents, shape (nwl,) (PROSPECT-PRO)
        phi: Fluorescence quantum yield spectrum, shape (nwlF,)
    """

    nr: NDArray[np.float64]
    Kab: NDArray[np.float64]
    Kca: NDArray[np.float64]
    KcaV: Optional[NDArray[np.float64]] = None
    KcaZ: Optional[NDArray[np.float64]] = None
    Kw: NDArray[np.float64] = None
    Kdm: NDArray[np.float64] = None
    Ks: NDArray[np.float64] = None
    Kant: NDArray[np.float64] = None
    Kp: Optional[NDArray[np.float64]] = None
    Kcbc: Optional[NDArray[np.float64]] = None
    phi: NDArray[np.float64] = None


@dataclass
class LeafOpticalOutput:
    """Output from Fluspect leaf optical properties model.

    Attributes:
        refl: Leaf reflectance spectrum, shape (nwl,)
        tran: Leaf transmittance spectrum, shape (nwl,)
        kChlrel: Relative chlorophyll absorption, shape (nwl,)
        kCarrel: Relative carotenoid absorption, shape (nwl,)
        Mb: Backward fluorescence matrix, shape (nwlF, nwlE)
        Mf: Forward fluorescence matrix, shape (nwlF, nwlE)
    """

    refl: NDArray[np.float64]
    tran: NDArray[np.float64]
    kChlrel: NDArray[np.float64]
    kCarrel: Optional[NDArray[np.float64]] = None
    Mb: Optional[NDArray[np.float64]] = None
    Mf: Optional[NDArray[np.float64]] = None


def calctav(alfa: float, nr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate average transmittance at interface.

    Computes the transmittance averaged over all azimuth angles for
    a given incidence angle, accounting for Fresnel reflections.

    Args:
        alfa: Incidence angle [degrees]
        nr: Refractive index array, shape (nwl,)

    Returns:
        Average transmittance, shape (nwl,)
    """
    rd = np.pi / 180
    n2 = nr ** 2
    np_ = n2 + 1
    nm = n2 - 1
    a = (nr + 1) ** 2 / 2
    k = -(n2 - 1) ** 2 / 4
    sa = np.sin(alfa * rd)

    if alfa != 90:
        b1 = np.sqrt((sa ** 2 - np_ / 2) ** 2 + k)
    else:
        b1 = np.zeros_like(nr)

    b2 = sa ** 2 - np_ / 2
    b = b1 - b2
    b3 = b ** 3
    a3 = a ** 3

    ts = (k ** 2 / (6 * b3) + k / b - b / 2) - (k ** 2 / (6 * a3) + k / a - a / 2)

    tp1 = -2 * n2 * (b - a) / (np_ ** 2)
    tp2 = -2 * n2 * np_ * np.log(b / a) / (nm ** 2)
    tp3 = n2 * (1 / b - 1 / a) / 2
    tp4 = 16 * n2 ** 2 * (n2 ** 2 + 1) * np.log((2 * np_ * b - nm ** 2) / (2 * np_ * a - nm ** 2)) / (np_ ** 3 * nm ** 2)
    tp5 = 16 * n2 ** 3 * (1 / (2 * np_ * b - nm ** 2) - 1 / (2 * np_ * a - nm ** 2)) / (np_ ** 3)

    tp = tp1 + tp2 + tp3 + tp4 + tp5
    tav = (ts + tp) / (2 * sa ** 2)

    return tav


def fluspect(
    leafbio,
    optipar: OptiPar,
    wlP: NDArray[np.float64],
    wlE: Optional[NDArray[np.float64]] = None,
    wlF: Optional[NDArray[np.float64]] = None,
    ndub: int = 15,
    resolution: int = 5,
) -> LeafOpticalOutput:
    """Calculate leaf optical properties using Fluspect model.

    Computes leaf reflectance, transmittance, and fluorescence matrices
    based on leaf biochemistry and PROSPECT-derived optical parameters.

    Args:
        leafbio: LeafBio dataclass with Cab, Cca, Cdm, Cw, Cs, Cant, Cp, Cbc, N, fqe
        optipar: OptiPar with absorption coefficients and refractive index
        wlP: PROSPECT wavelengths, shape (nwlP,) [nm]
        wlE: Excitation wavelengths for fluorescence, shape (nwlE,) [nm]
        wlF: Fluorescence emission wavelengths, shape (nwlF,) [nm]
        ndub: Number of doublings for fluorescence calculation (default 15)
        resolution: Spectral resolution reduction factor (default 5)

    Returns:
        LeafOpticalOutput with refl, tran, kChlrel, kCarrel, Mb, Mf
    """
    # Extract leaf parameters
    Cab = leafbio.Cab
    Cca = leafbio.Cca
    Cdm = leafbio.Cdm
    Cw = leafbio.Cw
    Cs = leafbio.Cs
    Cant = getattr(leafbio, "Cant", 0.0)
    Cp = getattr(leafbio, "Cp", 0.0)
    Cbc = getattr(leafbio, "Cbc", 0.0)
    N = leafbio.N
    fqe = leafbio.fqe
    V2Z = getattr(leafbio, "V2Z", -999)

    # Get optical parameters
    nr = optipar.nr
    Kab = optipar.Kab
    Kdm = optipar.Kdm if optipar.Kdm is not None else np.zeros_like(Kab)
    Kw = optipar.Kw if optipar.Kw is not None else np.zeros_like(Kab)
    Ks = optipar.Ks if optipar.Ks is not None else np.zeros_like(Kab)
    Kant = optipar.Kant if optipar.Kant is not None else np.zeros_like(Kab)
    Kp = optipar.Kp if optipar.Kp is not None else np.zeros_like(Kab)
    Kcbc = optipar.Kcbc if optipar.Kcbc is not None else np.zeros_like(Kab)

    # Handle carotenoid absorption based on V2Z
    if V2Z == -999:
        Kca = optipar.Kca
    else:
        # Linear combination of violaxanthin and zeaxanthin
        KcaV = optipar.KcaV if optipar.KcaV is not None else optipar.Kca
        KcaZ = optipar.KcaZ if optipar.KcaZ is not None else optipar.Kca
        Kca = (1 - V2Z) * KcaV + V2Z * KcaZ

    # Total absorption coefficient
    Kall = (Cab * Kab + Cca * Kca + Cdm * Kdm + Cw * Kw + Cs * Ks +
            Cant * Kant + Cp * Kp + Cbc * Kcbc) / N

    # Calculate transmittance using exponential integral
    j = Kall > 0
    t1 = (1 - Kall) * np.exp(-Kall)
    t2 = Kall ** 2 * (-expint(-Kall))  # Note: scipy expint is Ei, need -Ei(-x)
    tau = np.ones_like(t1)
    tau[j] = t1[j] + t2[j]

    # Relative chlorophyll and carotenoid absorption
    kChlrel = np.zeros_like(t1)
    kCarrel = np.zeros_like(t1)
    kChlrel[j] = Cab * Kab[j] / (Kall[j] * N)
    kCarrel[j] = Cca * Kca[j] / (Kall[j] * N)

    # Interface reflectances and transmittances
    talf = calctav(59, nr)
    ralf = 1 - talf
    t12 = calctav(90, nr)
    r12 = 1 - t12
    t21 = t12 / nr ** 2
    r21 = 1 - t21

    # Top surface
    denom = 1 - r21 * r21 * tau ** 2
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta

    # Bottom surface
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    # Stokes equations for N-1 layers
    D = np.sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t))
    rq = r ** 2
    tq = t ** 2
    a = (1 + rq - tq + D) / (2 * r)
    b = (1 - rq + tq + D) / (2 * t)

    bNm1 = b ** (N - 1)
    bN2 = bNm1 ** 2
    a2 = a ** 2
    denom = a2 * bN2 - 1
    Rsub = a * (bN2 - 1) / denom
    Tsub = bNm1 * (a2 - 1) / denom

    # Case of zero absorption
    j = (r + t) >= 1
    Tsub[j] = t[j] / (t[j] + (1 - t[j]) * (N - 1))
    Rsub[j] = 1 - Tsub[j]

    # Final reflectance and transmittance
    denom = 1 - Rsub * r
    tran = Ta * Tsub / denom
    refl = Ra + Ta * Rsub * t / denom

    # Create output
    output = LeafOpticalOutput(
        refl=refl,
        tran=tran,
        kChlrel=kChlrel,
        kCarrel=kCarrel,
    )

    # Fluorescence calculation (if fqe > 0)
    if fqe > 0 and wlE is not None and wlF is not None and optipar.phi is not None:
        # Calculate fluorescence matrices
        Mb, Mf = _calculate_fluorescence(
            refl, tran, kChlrel, ralf, talf, r21, t21,
            wlP, wlE, wlF, optipar.phi, fqe, ndub, resolution
        )
        output.Mb = Mb
        output.Mf = Mf

    return output


def _calculate_fluorescence(
    refl: NDArray[np.float64],
    tran: NDArray[np.float64],
    kChlrel: NDArray[np.float64],
    ralf: NDArray[np.float64],
    talf: NDArray[np.float64],
    r21: NDArray[np.float64],
    t21: NDArray[np.float64],
    wlP: NDArray[np.float64],
    wlE: NDArray[np.float64],
    wlF: NDArray[np.float64],
    phi: NDArray[np.float64],
    fqe: float,
    ndub: int,
    resolution: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate fluorescence excitation-emission matrices.

    Internal function that performs the doubling method to calculate
    forward and backward fluorescence matrices.

    This follows MATLAB fluspect_B_CX.m exactly:
    1. Extract the mesophyll layer by removing leaf-air interfaces
    2. Derive Kubelka-Munk k and s from mesophyll layer
    3. Perform doubling method for fluorescence
    4. Add interfaces back at the end

    Args:
        refl, tran: Leaf reflectance and transmittance
        kChlrel: Relative chlorophyll absorption
        ralf, talf, r21, t21: Interface properties
        wlP: PROSPECT wavelengths
        wlE, wlF: Excitation and fluorescence wavelengths
        phi: Fluorescence quantum yield spectrum
        fqe: Fluorescence quantum efficiency
        ndub: Number of doublings
        resolution: Spectral resolution factor

    Returns:
        Tuple of (Mb, Mf) fluorescence matrices
    """
    # Reduce resolution for excitation wavelengths (matching MATLAB)
    wle = np.arange(400, 751, resolution).astype(float)
    wlf = np.arange(640, 851, 4).astype(float)

    # =========================================================================
    # Extract mesophyll layer by removing leaf-air interfaces
    # (MATLAB fluspect_B_CX.m lines 166-172)
    # =========================================================================
    # Remove the top interface
    Rb = (refl - ralf) / (talf * t21 + (refl - ralf) * r21)
    # Derive Z from the transmittance
    Z = tran * (1 - Rb * r21) / (talf * t21)
    # Reflectance and transmittance of the leaf mesophyll layer
    rho = (Rb - r21 * Z ** 2) / (1 - (r21 * Z) ** 2)
    tau = (1 - Rb * r21) / (1 - (r21 * Z) ** 2) * Z
    # Use t and r for mesophyll layer
    t = tau
    r = np.maximum(rho, 0)  # Avoid negative r

    # =========================================================================
    # Derive Kubelka-Munk s and k from mesophyll layer
    # (MATLAB fluspect_B_CX.m lines 174-192)
    # =========================================================================
    # Initialize arrays
    D = np.zeros_like(r)
    a = np.ones_like(r)
    b = np.ones_like(r)

    # Normal case where r+t < 1 (absorption present)
    I_rt = (r + t) < 1
    D[I_rt] = np.sqrt((1 + r[I_rt] + t[I_rt]) *
                      (1 + r[I_rt] - t[I_rt]) *
                      (1 - r[I_rt] + t[I_rt]) *
                      (1 - r[I_rt] - t[I_rt]))

    # Avoid division by zero
    r_safe = np.where(r > 1e-10, r, 1e-10)
    t_safe = np.where(t > 1e-10, t, 1e-10)

    a[I_rt] = (1 + r[I_rt] ** 2 - t[I_rt] ** 2 + D[I_rt]) / (2 * r_safe[I_rt])
    b[I_rt] = (1 - r[I_rt] ** 2 + t[I_rt] ** 2 + D[I_rt]) / (2 * t_safe[I_rt])

    # Calculate s and k
    s_full = r / t_safe  # Default: s = r/t
    I_a = (a > 1) & np.isfinite(a)  # Cases where a > 1 and not Inf
    b_safe = np.where(b > 1e-10, b, 1e-10)
    s_full[I_a] = 2 * a[I_a] / (a[I_a] ** 2 - 1) * np.log(b_safe[I_a])

    k_full = np.log(b_safe)  # Default: k = log(b)
    k_full[I_a] = (a[I_a] - 1) / (a[I_a] + 1) * np.log(b_safe[I_a])

    # Chlorophyll-specific absorption
    kChl_full = kChlrel * k_full

    # =========================================================================
    # Interpolate to reduced wavelength grids
    # =========================================================================
    k_iwle = np.interp(wle, wlP, k_full)
    s_iwle = np.interp(wle, wlP, s_full)
    kChl_iwle = np.interp(wle, wlP, kChl_full)
    r21_iwle = np.interp(wle, wlP, r21)
    rho_iwle = np.interp(wle, wlP, rho)
    tau_iwle = np.interp(wle, wlP, tau)
    talf_iwle = np.interp(wle, wlP, talf)

    # Find indices of fluorescence wavelengths in wlP
    Iwlf = np.searchsorted(wlP, wlf)
    Iwlf = np.clip(Iwlf, 0, len(wlP) - 1)

    # Get fluorescence phi at emission wavelengths
    phi_f = np.interp(wlf, wlP, phi)

    eps = 2 ** (-ndub)

    # =========================================================================
    # Initialize for doubling
    # (MATLAB fluspect_B_CX.m lines 215-222)
    # =========================================================================
    te = 1 - (k_iwle + s_iwle) * eps
    tf = 1 - (k_full[Iwlf] + s_full[Iwlf]) * eps
    re = s_iwle * eps
    rf = s_full[Iwlf] * eps

    # Sigmoid function for wavelength-dependent transition
    sigmoid = 1 / (1 + np.exp(-wlf[:, np.newaxis] / 10) * np.exp(wle[np.newaxis, :] / 10))

    # Initialize fluorescence matrices
    Mf = resolution * fqe * (0.5 * phi_f[:, np.newaxis] * eps) * kChl_iwle[np.newaxis, :] * sigmoid
    Mb = Mf.copy()

    nwle = len(wle)
    nwlf = len(wlf)

    # Row and column of ones for broadcasting
    Ih = np.ones((1, nwle))
    Iv = np.ones((nwlf, 1))

    # =========================================================================
    # Doubling routine
    # (MATLAB fluspect_B_CX.m lines 229-242)
    # =========================================================================
    for _ in range(ndub):
        xe = te / (1 - re * re)
        ten = te * xe
        ren = re * (1 + ten)

        xf = tf / (1 - rf * rf)
        tfn = tf * xf
        rfn = rf * (1 + tfn)

        # Calculate coupling matrices
        A11 = xf[:, np.newaxis] * Ih + Iv * xe[np.newaxis, :]
        A12 = (xf[:, np.newaxis] * xe[np.newaxis, :]) * (rf[:, np.newaxis] * Ih + Iv * re[np.newaxis, :])
        A21 = 1 + (xf[:, np.newaxis] * xe[np.newaxis, :]) * (1 + rf[:, np.newaxis] * re[np.newaxis, :])
        A22 = (xf * rf)[:, np.newaxis] * Ih + Iv * (xe * re)[np.newaxis, :]

        Mfn = Mf * A11 + Mb * A12
        Mbn = Mb * A21 + Mf * A22

        te = ten
        re = ren
        tf = tfn
        rf = rfn
        Mf = Mfn
        Mb = Mbn

    # =========================================================================
    # Add leaf-air interfaces back
    # (MATLAB fluspect_B_CX.m lines 244-264)
    # =========================================================================
    g = Mb
    f = Mf

    # Calculate Rb using mesophyll properties (rho, tau)
    Rb = rho + tau ** 2 * r21 / (1 - rho * r21)
    Rb_iwle = np.interp(wle, wlP, Rb)

    Xe = Iv * (talf_iwle / (1 - r21_iwle * Rb_iwle))[np.newaxis, :]
    Xf = (t21[Iwlf] / (1 - r21[Iwlf] * Rb[Iwlf]))[:, np.newaxis] * Ih
    Ye = Iv * (tau_iwle * r21_iwle / (1 - rho_iwle * r21_iwle))[np.newaxis, :]
    Yf = (tau[Iwlf] * r21[Iwlf] / (1 - rho[Iwlf] * r21[Iwlf]))[:, np.newaxis] * Ih

    A = Xe * (1 + Ye * Yf) * Xf
    B = Xe * (Ye + Yf) * Xf

    Mb_final = A * g + B * f
    Mf_final = A * f + B * g

    return Mb_final, Mf_final


def prospect(
    leafbio,
    optipar: OptiPar,
    wlP: NDArray[np.float64],
) -> LeafOpticalOutput:
    """Run PROSPECT model without fluorescence.

    Convenience function that calls fluspect with fqe=0.

    Args:
        leafbio: LeafBio dataclass with leaf parameters
        optipar: OptiPar with absorption coefficients
        wlP: PROSPECT wavelengths [nm]

    Returns:
        LeafOpticalOutput with refl and tran only
    """
    # Create a copy of leafbio with fqe=0
    from dataclasses import replace
    leafbio_nofluor = replace(leafbio, fqe=0.0) if hasattr(leafbio, "__dataclass_fields__") else leafbio

    return fluspect(leafbio_nofluor, optipar, wlP)
