"""Biochemical photosynthesis model with numba acceleration.

Translated from: src/fluxes/biochemical.m

Implements the Farquhar-von Caemmerer-Berry (FvCB) C3/C4 photosynthesis model
with Ball-Berry stomatal conductance and chlorophyll fluorescence.

All core calculations use numba JIT compilation for performance.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from numba import jit

from ..constants import CONSTANTS, TEMP_RESPONSE


# ============================================================================
# Constants
# ============================================================================
R = 8.314  # Gas constant [J mol-1 K-1]
Tref = 298.15  # Reference temperature (25°C in K)
Kc25 = 405e-6  # Rubisco Km for CO2 [bar]
Ko25 = 279e-3  # Rubisco Km for O2 [bar]
spfy25 = 2444  # Specificity factor
rhoa = 1.2047  # Air density [kg m-3]
Mair = 28.96  # Molecular mass of air [g mol-1]


# ============================================================================
# Output dataclass
# ============================================================================
@dataclass
class BiochemicalOutput:
    """Output from the biochemical photosynthesis model.

    Attributes:
        A: Net assimilation rate [µmol m-2 s-1]
        Ag: Gross assimilation rate [µmol m-2 s-1]
        Ci: Internal CO2 concentration [ppm]
        gs: Stomatal conductance [mol m-2 s-1]
        rcw: Stomatal resistance [s m-1]
        Vcmax: Carboxylation capacity after temperature correction [µmol m-2 s-1]
        Rd: Dark respiration [µmol m-2 s-1]
        Ja: Actual electron transport rate [µmol m-2 s-1]
        ps: Photochemical yield [-]
        eta: Fluorescence efficiency (fs/fo0) [-]
        fs: Steady-state fluorescence yield [-]
        fo: Light-adapted Fo [-]
        fm: Light-adapted Fm [-]
        fo0: Dark-adapted Fo [-]
        fm0: Dark-adapted Fm [-]
        Kn: NPQ rate constant [-]
        NPQ: Non-photochemical quenching [-]
        qE: Energy-dependent quenching [-]
        qQ: Photochemical quenching [-]
        SIF: Fluorescence flux [relative units]
    """

    A: Union[float, NDArray[np.float64]] = 0.0
    Ag: Union[float, NDArray[np.float64]] = 0.0
    Ci: Union[float, NDArray[np.float64]] = 0.0
    gs: Union[float, NDArray[np.float64]] = 0.0
    rcw: Union[float, NDArray[np.float64]] = 0.0
    Vcmax: Union[float, NDArray[np.float64]] = 0.0
    Rd: Union[float, NDArray[np.float64]] = 0.0
    Ja: Union[float, NDArray[np.float64]] = 0.0
    ps: Union[float, NDArray[np.float64]] = 0.0
    eta: Union[float, NDArray[np.float64]] = 0.0
    fs: Union[float, NDArray[np.float64]] = 0.0
    fo: Union[float, NDArray[np.float64]] = 0.0
    fm: Union[float, NDArray[np.float64]] = 0.0
    fo0: Union[float, NDArray[np.float64]] = 0.0
    fm0: Union[float, NDArray[np.float64]] = 0.0
    Kn: Union[float, NDArray[np.float64]] = 0.0
    NPQ: Union[float, NDArray[np.float64]] = 0.0
    qE: Union[float, NDArray[np.float64]] = 0.0
    qQ: Union[float, NDArray[np.float64]] = 0.0
    SIF: Union[float, NDArray[np.float64]] = 0.0


# ============================================================================
# Numba-compiled helper functions
# ============================================================================
@jit(nopython=True, cache=True)
def temperature_correction(T_K, deltaHa):
    """Arrhenius temperature function for C3 photosynthesis."""
    return np.exp(deltaHa / (Tref * R) * (1 - Tref / T_K))


@jit(nopython=True, cache=True)
def high_temp_inhibition(T_K, deltaS, deltaHd):
    """High temperature inhibition function for C3 photosynthesis."""
    num = 1 + np.exp((Tref * deltaS - deltaHd) / (Tref * R))
    denom = 1 + np.exp((deltaS * T_K - deltaHd) / (R * T_K))
    return num / denom


@jit(nopython=True, cache=True)
def satvap_numba(T):
    """Saturation vapor pressure [hPa] from temperature [°C]."""
    return 6.107 * 10.0 ** (7.5 * T / (237.3 + T))


@jit(nopython=True, cache=True)
def brentq_numba(a, b, args, tol=1e-7, maxiter=100):
    """Fixed-point solver matching MATLAB's fixedp_brent_ari.

    Finds Ci where Ci_computed(Ci) = Ci (fixed point).
    """
    # Unpack args
    (Vcmax, Gamma_star, MM_consts, Je, effcon, atheta,
     Vs_C3, Rd, ppm2bar, BallBerrySlope, BallBerry0,
     Cs_bar, RH, min_Ci) = args

    def compute_next_ci(Ci):
        """Compute A from Ci, then compute next Ci from Ball-Berry."""
        Vc = Vcmax * (Ci - Gamma_star) / (MM_consts + Ci)
        CO2_pe = (Ci - Gamma_star) / (Ci + 2 * Gamma_star + 1e-10) * effcon
        Ve = Je * CO2_pe

        discr = max(0, (Vc + Ve)**2 - 4 * atheta * Vc * Ve)
        V = (Vc + Ve - np.sqrt(discr)) / (2 * atheta)
        discr2 = max(0, (V + Vs_C3)**2 - 4 * 0.98 * V * Vs_C3)
        Ag = (V + Vs_C3 - np.sqrt(discr2)) / (2 * 0.98)
        A = Ag - Rd

        A_bar = A * ppm2bar
        gs = max(BallBerry0, BallBerrySlope * A_bar * RH / (Cs_bar + 1e-9) + BallBerry0)
        Ci_next = max(min_Ci * Cs_bar, Cs_bar - 1.6 * A_bar / gs)
        return Ci_next

    def f(Ci):
        return compute_next_ci(Ci) - Ci

    # MATLAB approach: start from Cs
    a = Cs_bar
    Ci_next_a = compute_next_ci(a)
    err_a = Ci_next_a - a
    b = Ci_next_a
    err_b = f(b)

    if abs(err_b) < tol:
        return b

    if err_a * err_b < 0:
        fa, fb = err_a, err_b
    else:
        # Try secant extrapolation
        if abs(err_b - err_a) > 1e-15:
            x1 = b - err_b * (b - a) / (err_b - err_a)
            x1 = max(0.0, min(2.0 * Cs_bar, x1))
            err_x1 = f(x1)

            if err_x1 * err_a < 0:
                if abs(err_b) < abs(err_a):
                    a, err_a = b, err_b
                b, err_b = x1, err_x1
            elif err_x1 * err_b < 0:
                a, err_a = b, err_b
                b, err_b = x1, err_x1

        # Walking out if still not bracketed
        if err_a * err_b > 0:
            if abs(err_b) < abs(err_a):
                a, b = b, a
                err_a, err_b = err_b, err_a

            if err_a > 0 and err_b > 0:
                for _ in range(10):
                    diff_ab = b - a
                    a = a - diff_ab
                    a = max(0.0, a)
                    err_a = f(a)
                    if err_a * err_b < 0:
                        break
                    if a <= 0:
                        break
            elif err_a < 0 and err_b < 0:
                a = 0.0
                err_a = f(a)

        # Fallback: damped fixed-point iteration
        if err_a * err_b > 0:
            Ci = Cs_bar
            damping = 0.5
            for _ in range(50):
                Ci_old = Ci
                Ci_new = compute_next_ci(Ci)
                Ci = damping * Ci_new + (1 - damping) * Ci_old
                if abs(Ci - Ci_old) < tol:
                    break
            return Ci

        fa, fb = err_a, err_b

    # Brent's method
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    e = d

    for _ in range(maxiter):
        if abs(fb) < tol:
            return b

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol1 = 2 * 2.2e-16 * abs(b) + tol / 2
        m = (c - b) / 2

        if abs(m) <= tol1 or fb == 0:
            return b

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2 * m * s
                q = 1 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)

            if p > 0:
                q = -q
            else:
                p = -p

            if 2 * p < min(3 * m * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        a = b
        fa = fb

        if abs(d) > tol1:
            b = b + d
        elif m > 0:
            b = b + tol1
        else:
            b = b - tol1

        fb = f(b)

        if fb * fc > 0:
            c = a
            fc = fa
            d = b - a
            e = d

    return b


# ============================================================================
# Core biochemical calculation (numba-compiled)
# ============================================================================
@jit(nopython=True, cache=True)
def biochemical_core(
    Q, T, Cs, eb, p, O,
    Vcmax25, BallBerrySlope, BallBerry0, RdPerVcmax25,
    Kn0, Knalpha, Knbeta,
    delHaV, delSV, delHdV,
    delHaR, delSR, delHdR,
    delHaKc, delHaKo, delHaT,
    stressfactor=1.0,
):
    """Core biochemical calculation - numba optimized.

    Args:
        Q: Absorbed PAR [µmol m-2 s-1]
        T: Temperature [°C or K]
        Cs: CO2 at leaf surface [ppm]
        eb: Vapor pressure [hPa]
        p: Air pressure [hPa]
        O: O2 concentration [per mille]
        Vcmax25: Max carboxylation at 25°C [µmol m-2 s-1]
        BallBerrySlope: Ball-Berry slope
        BallBerry0: Ball-Berry intercept [mol m-2 s-1]
        RdPerVcmax25: Rd as fraction of Vcmax25
        Kn0, Knalpha, Knbeta: NPQ parameters
        delHa*, delS*, delHd*: Temperature response parameters
        stressfactor: Stress factor (0-1)

    Returns:
        Tuple: (A, Ag, rcw, Ci_ppm, Ja, eta, Kn)
    """
    # Convert temperature
    T_K = T + 273.15 if T < 200 else T
    T_C = T_K - 273.15

    # Pressure conversion
    ppm2bar = 1e-6 * (p * 1e-3)
    Cs_bar = Cs * ppm2bar
    O_bar = (O * 1e-3) * (p * 1e-3)

    # CO2 compensation point
    Gamma_star25 = 0.5 * O_bar / spfy25
    Rd25 = RdPerVcmax25 * Vcmax25

    # Temperature corrections (C3)
    fTv = temperature_correction(T_K, delHaV)
    fHTv = high_temp_inhibition(T_K, delSV, delHdV)
    Vcmax = Vcmax25 * fTv * fHTv * stressfactor

    fTr = temperature_correction(T_K, delHaR)
    fHTr = high_temp_inhibition(T_K, delSR, delHdR)
    Rd = Rd25 * fTr * fHTr * stressfactor

    Kc = Kc25 * temperature_correction(T_K, delHaKc)
    Ko = Ko25 * temperature_correction(T_K, delHaKo)
    Gamma_star = Gamma_star25 * temperature_correction(T_K, delHaT)

    # Fluorescence rate constants
    Kf = 0.05
    Kd = max(0.8738, 0.0301 * T_C + 0.0773)
    Kp = 4.0

    # Dark photochemistry fraction and potential electron transport
    po0 = Kp / (Kf + Kd + Kp)
    Je = 0.5 * po0 * Q

    # Relative humidity
    es = satvap_numba(T_C)
    RH = min(1.0, eb / es)

    # Michaelis-Menten constants (C3)
    MM_consts = Kc * (1 + O_bar / Ko)
    Vs_C3 = Vcmax / 2
    effcon = 0.2  # 1/5 for C3
    atheta = 0.8

    # Ball-Berry iteration
    min_Ci = 0.3
    if BallBerry0 == 0:
        Ci = max(min_Ci * Cs_bar, Cs_bar * (1 - 1.6 / (BallBerrySlope * RH + 1e-10)))
    else:
        args = (Vcmax, Gamma_star, MM_consts, Je, effcon, atheta,
                Vs_C3, Rd, ppm2bar, BallBerrySlope, BallBerry0,
                Cs_bar, RH, min_Ci)
        Ci = brentq_numba(0, 0, args, tol=1e-7, maxiter=100)

    # Final assimilation
    Vc = Vcmax * (Ci - Gamma_star) / (MM_consts + Ci)
    CO2_pe = (Ci - Gamma_star) / (Ci + 2 * Gamma_star + 1e-10) * effcon
    Ve = Je * CO2_pe

    discr = max(0, (Vc + Ve)**2 - 4 * atheta * Vc * Ve)
    V = (Vc + Ve - np.sqrt(discr)) / (2 * atheta)
    discr2 = max(0, (V + Vs_C3)**2 - 4 * 0.98 * V * Vs_C3)
    Ag = (V + Vs_C3 - np.sqrt(discr2)) / (2 * 0.98)
    A = Ag - Rd

    # Stomatal resistance
    gs = max(0.0, 1.6 * A * ppm2bar / (Cs_bar - Ci + 1e-10))
    if gs > 0:
        rcw = (rhoa / (Mair * 1e-3)) / gs
    else:
        rcw = np.inf

    # Electron transport and fluorescence
    Ja = Ag / max(CO2_pe, 1e-10)
    ps = po0 * Ja / max(Je, 1e-10) if Je > 1e-10 else po0
    x = max(0, 1 - ps / po0)
    x_alpha = x ** Knalpha
    Kn = Kn0 * (1 + Knbeta) * x_alpha / (Knbeta + x_alpha)

    fo0 = Kf / (Kf + Kp + Kd)
    fm = Kf / (Kf + Kd + Kn)
    fs = fm * (1 - ps)
    eta = fs / fo0

    Ci_ppm = Ci / ppm2bar
    return A, Ag, rcw, Ci_ppm, Ja, eta, Kn


# ============================================================================
# Heat flux calculations (numba-compiled)
# ============================================================================
@jit(nopython=True, cache=True)
def heatfluxes_core(ra, rs, Tc, ea, Ta, e_to_q, Ca, Ci):
    """Core heat flux calculation - numba optimized.

    Args:
        ra: Aerodynamic resistance [s m-1]
        rs: Stomatal resistance [s m-1]
        Tc: Surface temperature [°C]
        ea: Ambient vapor pressure [hPa]
        Ta: Air temperature [°C]
        e_to_q: Vapor pressure to specific humidity conversion
        Ca: Ambient CO2 concentration [ppm]
        Ci: Internal CO2 concentration [ppm]

    Returns:
        Tuple: (lE, H, ec, Cc, lambda_, s)
    """
    rhoa_local = 1.2047
    cp = 1004.0

    ei = satvap_numba(Tc)
    s = ei * 2.3026 * 7.5 * 237.3 / (237.3 + Tc) ** 2
    lambda_ = (2.501 - 0.002361 * Tc) * 1e6

    qi = ei * e_to_q
    qa = ea * e_to_q

    lE = rhoa_local / (ra + rs) * lambda_ * (qi - qa)
    H = (rhoa_local * cp) / ra * (Tc - Ta)
    ec = ea + (ei - ea) * ra / (ra + rs)
    Cc = Ca - (Ca - Ci) * ra / (ra + rs)

    return lE, H, ec, Cc, lambda_, s


# ============================================================================
# Vectorized functions for all layers (numba-compiled)
# ============================================================================
@jit(nopython=True, parallel=False, cache=True)
def biochemical_vectorized(
    Q, T, Cs, eb, p, O,
    Vcmax25_arr, BallBerrySlope, BallBerry0, RdPerVcmax25,
    Kn0, Knalpha, Knbeta,
    delHaV, delSV, delHdV,
    delHaR, delSR, delHdR,
    delHaKc, delHaKo, delHaT,
    stressfactor=1.0,
):
    """Vectorized biochemical for all layers.

    Args:
        Q, T, Cs, eb: Arrays of shape (nl,)
        Vcmax25_arr: Array of shape (nl,) with Vcmax25 * fV
        Other params: Scalars

    Returns:
        Tuple of arrays: (A, Ag, rcw, Ci, Ja, eta, Kn) - all shape (nl,)
    """
    nl = len(Q)
    A = np.zeros(nl)
    Ag = np.zeros(nl)
    rcw = np.zeros(nl)
    Ci = np.zeros(nl)
    Ja = np.zeros(nl)
    eta = np.zeros(nl)
    Kn = np.zeros(nl)

    for j in range(nl):
        A[j], Ag[j], rcw[j], Ci[j], Ja[j], eta[j], Kn[j] = biochemical_core(
            Q[j], T[j], Cs[j], eb[j], p, O,
            Vcmax25_arr[j], BallBerrySlope, BallBerry0, RdPerVcmax25,
            Kn0, Knalpha, Knbeta,
            delHaV, delSV, delHdV,
            delHaR, delSR, delHdR,
            delHaKc, delHaKo, delHaT,
            stressfactor,
        )

    return A, Ag, rcw, Ci, Ja, eta, Kn


@jit(nopython=True, parallel=False, cache=True)
def heatfluxes_vectorized(ra, rs_arr, Tc, ea, Ta, e_to_q, Ca, Ci_arr):
    """Vectorized heat fluxes for all layers.

    Args:
        ra: Scalar resistance
        rs_arr, Tc: Arrays of shape (nl,)
        ea, Ta, e_to_q, Ca: Scalars
        Ci_arr: Array of shape (nl,) - internal CO2 from biochemical

    Returns:
        Tuple of arrays: (lE, H, ec, Cc, lambda_, s) - all shape (nl,)
    """
    nl = len(Tc)
    lE = np.zeros(nl)
    H = np.zeros(nl)
    ec = np.zeros(nl)
    Cc = np.zeros(nl)
    lambda_ = np.zeros(nl)
    s = np.zeros(nl)

    for j in range(nl):
        lE[j], H[j], ec[j], Cc[j], lambda_[j], s[j] = heatfluxes_core(
            ra, rs_arr[j], Tc[j], ea, Ta, e_to_q, Ca, Ci_arr[j]
        )

    return lE, H, ec, Cc, lambda_, s


# ============================================================================
# Python wrapper for individual leaf (non-lite mode)
# ============================================================================
def biochemical_individual(
    Q: Union[float, NDArray[np.float64]],
    T: Union[float, NDArray[np.float64]],
    Cs: float,
    eb: float,
    p: float,
    O: float,
    Vcmax25: float,
    BallBerrySlope: float,
    BallBerry0: float,
    RdPerVcmax25: float,
    Type: Literal["C3", "C4"] = "C3",
    Kn0: float = 2.48,
    Knalpha: float = 2.83,
    Knbeta: float = 0.114,
    stressfactor: float = 1.0,
    apply_T_corr: bool = True,
    TDP: Optional[object] = None,
    g_m: float = float('inf'),
) -> BiochemicalOutput:
    """Calculate photosynthesis for a single leaf point.

    Wrapper that calls the numba-compiled core function and returns
    a BiochemicalOutput dataclass.

    Args:
        Q: Absorbed PAR flux [µmol m-2 s-1]
        T: Leaf temperature [K or °C]
        Cs: CO2 concentration at leaf surface [ppm]
        eb: Vapor pressure in boundary layer [hPa]
        p: Air pressure [hPa]
        O: O2 concentration [per mille]
        Vcmax25: Maximum carboxylation rate at 25°C [µmol m-2 s-1]
        BallBerrySlope: Ball-Berry slope parameter
        BallBerry0: Ball-Berry intercept [mol m-2 s-1]
        RdPerVcmax25: Respiration as fraction of Vcmax25
        Type: Photosynthetic pathway ('C3' or 'C4')
        Kn0: Maximum NPQ rate constant
        Knalpha: NPQ alpha parameter
        Knbeta: NPQ beta parameter
        stressfactor: Stress factor reducing Vcmax [0-1]
        apply_T_corr: Apply temperature corrections
        TDP: Temperature response parameters
        g_m: Mesophyll conductance [mol m-2 s-1 bar-1] (not used in lite)

    Returns:
        BiochemicalOutput with photosynthesis results
    """
    # Get temperature response parameters
    if TDP is None:
        TDP = TEMP_RESPONSE

    # Call numba core
    A, Ag, rcw, Ci, Ja, eta, Kn = biochemical_core(
        float(Q), float(T), float(Cs), float(eb), float(p), float(O),
        float(Vcmax25), float(BallBerrySlope), float(BallBerry0), float(RdPerVcmax25),
        float(Kn0), float(Knalpha), float(Knbeta),
        TDP.delHaV, TDP.delSV, TDP.delHdV,
        TDP.delHaR, TDP.delSR, TDP.delHdR,
        TDP.delHaKc, TDP.delHaKo, TDP.delHaT,
        float(stressfactor),
    )

    # Calculate additional fluorescence parameters for output
    T_K = T + 273.15 if T < 200 else T
    T_C = T_K - 273.15

    Kf = 0.05
    Kd = max(0.8738, 0.0301 * T_C + 0.0773)
    Kp = 4.0
    po0 = Kp / (Kf + Kd + Kp)
    Je = 0.5 * po0 * Q

    ps = po0 * Ja / max(Je, 1e-10) if Je > 1e-10 else po0
    x = max(0, 1 - ps / po0)

    fo0 = Kf / (Kf + Kp + Kd)
    fo = Kf / (Kf + Kp + Kd + Kn)
    fm = Kf / (Kf + Kd + Kn)
    fm0 = Kf / (Kf + Kd)
    fs = fm * (1 - ps)

    qQ = 1 - (fs - fo) / max(fm - fo, 1e-10)
    qE = 1 - (fm - fo) / max(fm0 - fo0, 1e-10)

    # Stomatal conductance
    ppm2bar = 1e-6 * (p * 1e-3)
    Cs_bar = Cs * ppm2bar
    Ci_bar = Ci * ppm2bar
    gs = max(0.0, 1.6 * A * ppm2bar / (Cs_bar - Ci_bar + 1e-10))

    return BiochemicalOutput(
        A=A,
        Ag=Ag,
        Ci=Ci,
        gs=gs,
        rcw=rcw,
        Ja=Ja,
        ps=ps,
        eta=eta,
        fs=fs,
        fo=fo,
        fm=fm,
        fo0=fo0,
        fm0=fm0,
        Kn=Kn,
        NPQ=Kn / (Kf + Kd),
        qE=qE,
        qQ=qQ,
        SIF=fs * Q,
    )
