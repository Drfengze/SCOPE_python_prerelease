"""Energy Balance Module (ebal).

Translated from: src/fluxes/ebal.m

This module implements the iterative energy balance solver that couples:
- Thermal radiative transfer (RTMt)
- Leaf biochemistry (photosynthesis, stomatal conductance)
- Aerodynamic physics (resistances)
- Heat flux calculations

The energy balance iteration loop works as follows:
1. RTMt: Thermal radiation emitted by vegetation
2. resistances: Aerodynamic and boundary layer resistances
3. biochemical: Photosynthesis, fluorescence, stomatal resistance
4. heatfluxes: Sensible and latent heat flux
5. Evaluate energy balance and adjust temperatures
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..constants import CONSTANTS, TEMP_RESPONSE
from ..supporting.meanleaf import meanleaf
from .biochemical import biochemical_vectorized, heatfluxes_vectorized


def monin_obukhov(constants, meteo, H, ustar):
    """Calculate Monin-Obukhov length for atmospheric stability.

    Translated from: src/supporting/Monin_Obukhov.m

    L > 0: stable (H < 0, surface cooling)
    L < 0: unstable (H > 0, surface heating)
    |L| large: neutral

    Args:
        constants: Constants with rhoa, cp, kappa, g
        meteo: Meteo with Ta
        H: Sensible heat flux [W m-2]
        ustar: Friction velocity [m s-1]

    Returns:
        L: Obukhov length [m]
    """
    rhoa = constants.rhoa
    cp = constants.cp
    kappa = constants.kappa
    g = constants.g if hasattr(constants, 'g') else 9.81
    Ta = meteo.Ta

    # L = -rhoa * cp * ustar^3 * (Ta + 273.15) / (kappa * g * H)
    with np.errstate(divide='ignore', invalid='ignore'):
        L = -rhoa * cp * ustar**3 * (Ta + 273.15) / (kappa * g * H)

    # Handle NaN (when H = 0)
    if np.isnan(L) or np.isinf(L):
        L = -1e6  # Default to slightly unstable (MATLAB default)

    return L


@dataclass
class EnergyBalanceOutput:
    """Output from energy balance calculation.

    Attributes:
        iter_count: Number of iterations to convergence
        converged: Whether energy balance converged

        # Temperature outputs
        Tcu: Sunlit leaf temperature [C] - shape (nl,) or (nli, nlazi, nl)
        Tch: Shaded leaf temperature per layer [C] - shape (nl,)
        Tsu: Sunlit soil temperature [C]
        Tsh: Shaded soil temperature [C]
        Tcave: Average canopy temperature [C]
        Tsave: Average soil temperature [C]

        # Canopy fluxes
        Rnctot: Canopy net radiation [W m-2]
        lEctot: Canopy latent heat [W m-2]
        Hctot: Canopy sensible heat [W m-2]
        Actot: Total assimilation [umol m-2 s-1]

        # Soil fluxes
        Rnstot: Soil net radiation [W m-2]
        lEstot: Soil latent heat [W m-2]
        Hstot: Soil sensible heat [W m-2]
        Gtot: Soil heat flux [W m-2]

        # Total fluxes
        Rntot: Total net radiation [W m-2]
        lEtot: Total latent heat [W m-2]
        Htot: Total sensible heat [W m-2]

        # Additional outputs
        bcu: Biochemical output for sunlit leaves
        bch: Biochemical output for shaded leaves
        resist: Resistance output
        canopy_emis: Effective canopy emissivity
    """
    iter_count: int = 0
    converged: bool = False

    Tcu: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Tch: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    Tsu: float = 25.0
    Tsh: float = 25.0
    Tcave: float = 25.0
    Tsave: float = 25.0

    Rnctot: float = 0.0
    lEctot: float = 0.0
    Hctot: float = 0.0
    Actot: float = 0.0

    Rnstot: float = 0.0
    lEstot: float = 0.0
    Hstot: float = 0.0
    Gtot: float = 0.0

    Rntot: float = 0.0
    lEtot: float = 0.0
    Htot: float = 0.0

    bcu: Optional[object] = None
    bch: Optional[object] = None
    resist: Optional[object] = None

    canopy_emis: float = 0.98


def es_fun(T):
    """Saturation vapor pressure [hPa] from temperature [C]."""
    return 6.107 * 10.0 ** (7.5 * T / (237.3 + T))


def s_fun(es, T):
    """Slope of saturation vapor pressure curve [hPa/C]."""
    return es * 2.3026 * 7.5 * 237.3 / (237.3 + T) ** 2


def aggregator(
    LAI: float,
    sunlit_flux: NDArray[np.float64],
    shaded_flux: NDArray[np.float64],
    Fs: NDArray[np.float64],
    canopy,
    integr: str,
) -> float:
    """Aggregate sunlit and shaded fluxes to canopy total.

    Translated from MATLAB aggregator function in ebal.m:
    flux_tot = LAI*(meanleaf(canopy,sunlit_flux,'angles_and_layers',Fs) +
                    meanleaf(canopy,shaded_flux,'layers',1-Fs))

    Args:
        LAI: Leaf area index
        sunlit_flux: Flux for sunlit leaves - shape (nl,) or (nli, nlazi, nl)
        shaded_flux: Flux for shaded leaves - shape (nl,)
        Fs: Sunlit fraction per layer - shape (nl,)
        canopy: Canopy structure with nlayers, nlincl, nlazi, lidf
        integr: Integration mode - 'angles' for full mode, 'layers' for lite mode

    Returns:
        Aggregated canopy flux (scalar)
    """
    lidf = canopy.lidf
    nlazi = canopy.nlazi
    nl = canopy.nlayers

    # Determine meanleaf choice based on integr and sunlit_flux shape
    if integr == 'angles' and sunlit_flux.ndim == 3:
        # Full angular mode: integrate over angles AND layers to get scalar
        sunlit_choice = 'angles_and_layers'
    else:
        # Lite mode or 1D flux: integrate over layers only
        sunlit_choice = 'layers'

    # Sunlit leaves: integrate to scalar
    sunlit_mean = meanleaf(sunlit_flux, lidf, nlazi, sunlit_choice, Fs[:nl])

    # Shaded leaves: always integrate over layers only (scalar output)
    shaded_mean = meanleaf(shaded_flux, lidf, nlazi, 'layers', 1 - Fs[:nl])

    flux_tot = LAI * (sunlit_mean + shaded_mean)

    return float(flux_tot)


def ebal(
    constants,
    options,
    rad,
    gap,
    meteo,
    soil,
    canopy,
    leafbio,
    k: int = 1,
    xyt: Optional[dict] = None,
    integr: str = 'angles',
) -> Tuple[int, object, dict, object, object, object, dict, dict, object]:
    """Solve energy balance iteratively.

    Translated from MATLAB ebal.m

    This function couples thermal RTM, biochemistry, and aerodynamic
    physics to find temperatures that close the energy balance.

    Args:
        constants: Physical constants
        options: Simulation options
        rad: Radiation structure from RTMo
        gap: Gap probabilities from RTMo
        meteo: Meteorological data
        soil: Soil properties
        canopy: Canopy structure
        leafbio: Leaf biochemical parameters
        k: Time step index (for soil heat)
        xyt: Time data (for soil heat)
        integr: Integration mode ('angles' or 'layers')

    Returns:
        Tuple of (iter_count, rad, thermal, soil, bcu, bch, fluxes, resist_out, meteo)
    """
    from .biochemical import biochemical_individual
    from .resistances import resistances
    from .heatfluxes import heatfluxes
    from ..rtm.rtmt import rtmt_sb

    # Parameters for closure loop
    counter = 0
    maxit = 100
    maxEBer = 1.0  # W/m2
    Wc = 1.0
    CONT = True

    # Constants
    MH2O = constants.MH2O
    Mair = constants.Mair
    rhoa = constants.rhoa
    cp = constants.cp
    sigmaSB = constants.sigmaSB

    # Input preparation
    nl = canopy.nlayers
    nli = canopy.nlincl
    nlazi = canopy.nlazi
    lidf = canopy.lidf
    GAM = soil.GAM if hasattr(soil, 'GAM') else 0.0
    Ps = gap.Ps
    kV = canopy.kV if hasattr(canopy, 'kV') else 0.0
    xl = canopy.xl
    LAI = canopy.LAI
    rss = soil.rss if hasattr(soil, 'rss') else 500.0

    # Determine if lite mode
    lite = options.get('lite', False) if isinstance(options, dict) else getattr(options, 'lite', False)

    # Soil heat method
    SoilHeatMethod = options.get('soil_heat_method', 2) if isinstance(options, dict) else getattr(options, 'soil_heat_method', 2)

    # Monin-Obukhov stability correction (critical for stable conditions)
    use_monin_obukhov = options.get('MoninObukhov', True) if isinstance(options, dict) else getattr(options, 'MoninObukhov', True)

    # Meteo
    Ta = meteo.Ta
    ea = meteo.ea
    Ca = meteo.Ca
    p = meteo.p

    # Get radiation arrays
    Rnuc_sw = rad.Rnuc  # Net shortwave for sunlit - may be (nl,) or (nli, nlazi, nl)
    Rnhc_sw = rad.Rnhc  # Net shortwave for shaded - (nl,)

    # Initial vapor pressure and CO2 at leaf boundary
    ech = ea * np.ones(nl)
    Cch = Ca * np.ones(nl)
    if lite:
        ecu = ea + 0 * Rnuc_sw
        Ccu = Ca + 0 * Rnuc_sw
    else:
        ecu = ea * np.ones((nli, nlazi, nl))
        Ccu = Ca * np.ones((nli, nlazi, nl))

    # Conversion factor
    e_to_q = MH2O / Mair / p

    # Sunlit fractions
    Fc = Ps[:nl]
    Fs_soil = np.array([1 - Ps[nl], Ps[nl]])  # [shaded, sunlit] soil fractions

    # Vertical profile of Vcmax
    fV = np.exp(kV * xl[:nl])

    # Initial temperatures
    Ts = np.array([Ta + 3, Ta + 3])  # Soil [shaded, sunlit]
    Tch = (Ta + 0.1) * np.ones(nl)  # Shaded leaves
    if lite:
        Tcu = (Ta + 0.3) * np.ones(nl)  # Sunlit leaves (lite mode)
        fVu = fV
    else:
        Tcu = (Ta + 0.3) * np.ones((nli, nlazi, nl))  # Sunlit leaves (full mode)
        fVu = np.zeros((nli, nlazi, nl))
        for j in range(nl):
            fVu[:, :, j] = fV[j]

    # Initial Monin-Obukhov length
    L = -1e6

    # Energy balance iteration
    while CONT:
        # 2.1 Net radiation including thermal
        # Call RTMt for thermal radiation
        thermal = rtmt_sb(
            rad=rad,
            soil=soil,
            leafbio=leafbio,
            canopy=canopy,
            gap=gap,
            Tcu=Tcu,
            Tch=Tch,
            Tsu=Ts[1],
            Tsh=Ts[0],
            constants=constants,
            Rli=meteo.Rli,
        )

        # Total net radiation
        Rnhc = Rnhc_sw + thermal.Rnhct  # Shaded leaves
        Rnuc = Rnuc_sw + thermal.Rnuct  # Sunlit leaves
        Rnhs = rad.Rnhs + thermal.Rnhst  # Shaded soil
        Rnus = rad.Rnus + thermal.Rnust  # Sunlit soil
        Rns = np.array([Rnhs, Rnus])

        # 2.3 Biochemical processes (numba-optimized)
        # Get temperature response parameters
        TDP = leafbio.TDP if hasattr(leafbio, 'TDP') and leafbio.TDP is not None else TEMP_RESPONSE

        # Prepare vectorized inputs for shaded leaves
        Q_h = rad.Pnh_Cab if np.ndim(rad.Pnh_Cab) > 0 else np.full(nl, rad.Pnh_Cab)
        Vcmax25_h = leafbio.Vcmax25 * fV

        # Shaded leaves - vectorized numba call
        bch_A, bch_Ag, bch_rcw, bch_Ci, bch_Ja, bch_eta, bch_Kn = biochemical_vectorized(
            Q_h, Tch, Cch, ech, p, 209.0,
            Vcmax25_h, leafbio.BallBerrySlope, leafbio.BallBerry0, leafbio.RdPerVcmax25,
            leafbio.Kn0, leafbio.Knalpha, leafbio.Knbeta,
            TDP.delHaV, TDP.delSV, TDP.delHdV,
            TDP.delHaR, TDP.delSR, TDP.delHdR,
            TDP.delHaKc, TDP.delHaKo, TDP.delHaT,
            leafbio.stressfactor,
        )

        # Sunlit leaves
        if lite:
            # Prepare vectorized inputs
            Q_u = rad.Pnu_Cab if np.ndim(rad.Pnu_Cab) > 0 else np.full(nl, rad.Pnu_Cab)
            Vcmax25_u = leafbio.Vcmax25 * fVu

            # Sunlit leaves - vectorized numba call
            bcu_A, bcu_Ag, bcu_rcw, bcu_Ci, bcu_Ja, bcu_eta, bcu_Kn = biochemical_vectorized(
                Q_u, Tcu, Ccu, ecu, p, 209.0,
                Vcmax25_u, leafbio.BallBerrySlope, leafbio.BallBerry0, leafbio.RdPerVcmax25,
                leafbio.Kn0, leafbio.Knalpha, leafbio.Knbeta,
                TDP.delHaV, TDP.delSV, TDP.delHdV,
                TDP.delHaR, TDP.delSR, TDP.delHdR,
                TDP.delHaKc, TDP.delHaKo, TDP.delHaT,
                leafbio.stressfactor,
            )
        else:
            # Full angular resolution
            bcu_A = np.zeros((nli, nlazi, nl))
            bcu_Ag = np.zeros((nli, nlazi, nl))
            bcu_rcw = np.zeros((nli, nlazi, nl))
            bcu_Ci = np.zeros((nli, nlazi, nl))
            bcu_Ja = np.zeros((nli, nlazi, nl))
            bcu_eta = np.zeros((nli, nlazi, nl))
            bcu_Kn = np.zeros((nli, nlazi, nl))

            # Get PAR for sunlit leaves (3D if available, otherwise expand from 1D)
            Pnu_Cab = rad.Pnu_Cab
            if Pnu_Cab.ndim == 1:
                # Expand 1D to 3D by replicating across all orientations
                Pnu_Cab = np.tile(Pnu_Cab[np.newaxis, np.newaxis, :], (nli, nlazi, 1))

            for j in range(nl):
                for i in range(nli):
                    for a in range(nlazi):
                        Q_val = Pnu_Cab[i, a, j] if Pnu_Cab.ndim == 3 else Pnu_Cab[j]
                        bc = biochemical_individual(
                            Q=Q_val,
                            T=Tcu[i, a, j] if Tcu.ndim == 3 else Tcu[j],
                            Cs=Ccu[i, a, j] if Ccu.ndim == 3 else Ccu[j],
                            eb=ecu[i, a, j] if ecu.ndim == 3 else ecu[j],
                            p=p,
                            O=209,
                            Vcmax25=leafbio.Vcmax25 * fVu[i, a, j],
                            BallBerrySlope=leafbio.BallBerrySlope,
                            BallBerry0=leafbio.BallBerry0,
                            RdPerVcmax25=leafbio.RdPerVcmax25,
                            Type=leafbio.Type if hasattr(leafbio, 'Type') else 'C3',
                        )
                        bcu_A[i, a, j] = bc.A
                        bcu_Ag[i, a, j] = bc.Ag
                        bcu_rcw[i, a, j] = bc.rcw
                        bcu_Ci[i, a, j] = bc.Ci
                        bcu_Ja[i, a, j] = bc.Ja
                        bcu_eta[i, a, j] = bc.eta
                        bcu_Kn[i, a, j] = bc.Kn

        # 2.4 Aerodynamic resistances
        resist_out = resistances(
            constants=constants,
            soil=soil,
            canopy=canopy,
            meteo=meteo,
        )
        ustar = resist_out.ustar
        raa = resist_out.raa
        rawc = resist_out.rawc
        raws = resist_out.raws
        rac = (LAI + 1) * (raa + rawc)
        ras = (LAI + 1) * (raa + raws)

        # 2.5 Heat fluxes (numba-optimized)
        # Shaded leaves - vectorized
        lEch, Hch, ech, Cch, lambdah, sh = heatfluxes_vectorized(rac, bch_rcw, Tch, ea, Ta, e_to_q, Ca, bch_Ci)

        # Sunlit leaves
        if lite:
            # Vectorized call
            lEcu, Hcu, ecu, Ccu, lambdau, su = heatfluxes_vectorized(rac, bcu_rcw, Tcu, ea, Ta, e_to_q, Ca, bcu_Ci)
        else:
            lEcu = np.zeros((nli, nlazi, nl))
            Hcu = np.zeros((nli, nlazi, nl))
            lambdau = np.zeros((nli, nlazi, nl))
            su = np.zeros((nli, nlazi, nl))
            for j in range(nl):
                for i in range(nli):
                    for a in range(nlazi):
                        hf = heatfluxes(rac, bcu_rcw[i, a, j], Tcu[i, a, j], ea, Ta, e_to_q, Ca, bcu_Ci[i, a, j], constants, es_fun, s_fun)
                        lEcu[i, a, j] = hf['lE']
                        Hcu[i, a, j] = hf['H']
                        ecu[i, a, j] = hf['ec']
                        Ccu[i, a, j] = hf['Cc']
                        lambdau[i, a, j] = hf['lambda_']
                        su[i, a, j] = hf['s']

        # Soil heat fluxes
        lEs = np.zeros(2)
        Hs = np.zeros(2)
        lambdas = np.zeros(2)
        ss = np.zeros(2)
        for i in range(2):
            hf = heatfluxes(ras, rss, Ts[i], ea, Ta, e_to_q, Ca, Ca, constants, es_fun, s_fun)
            lEs[i] = hf['lE']
            Hs[i] = hf['H']
            lambdas[i] = hf['lambda_']
            ss[i] = hf['s']

        # 2.6 Integration and total fluxes
        Hstot = np.dot(Fs_soil, Hs)
        Hctot = aggregator(LAI, Hcu, Hch, Fc, canopy, integr)
        Htot = Hstot + Hctot

        # Monin-Obukhov stability correction (critical for stable/unstable conditions)
        # Updates meteo.L which affects resistance calculation in next iteration
        if use_monin_obukhov:
            meteo.L = monin_obukhov(constants, meteo, Htot, ustar)

        # Ground heat flux
        if SoilHeatMethod == 2:
            G = 0.35 * Rns
            rs_thermal_val = soil.rs_thermal if hasattr(soil, 'rs_thermal') else 0.05
            dG = 4 * (1 - rs_thermal_val) * sigmaSB * (Ts + 273.15) ** 3 * 0.35
        elif SoilHeatMethod == 3:
            G = 0 * Rns
            dG = 0 * Rns
        else:
            G = 0.35 * Rns  # Default
            dG = 4 * 0.95 * sigmaSB * (Ts + 273.15) ** 3 * 0.35

        # 2.7 Energy balance errors
        EBerch = Rnhc - lEch - Hch
        EBercu = Rnuc - lEcu - Hcu
        EBers = Rns - lEs - Hs - G

        counter += 1
        maxEBercu = np.max(np.abs(EBercu))
        maxEBerch = np.max(np.abs(EBerch))
        maxEBers = np.max(np.abs(EBers))

        CONT = (maxEBercu > maxEBer or maxEBerch > maxEBer or maxEBers > maxEBer) and counter < maxit + 1

        if not CONT:
            if np.any(np.isnan([maxEBercu, maxEBerch, maxEBers])):
                print(f'WARNING: NaN in fluxes, counter = {counter}')
            break

        # Relaxation factor adjustment
        if counter == 10:
            Wc = 0.8
        if counter == 20:
            Wc = 0.6

        # 2.8 Update temperatures
        emis_leaf = leafbio.emis if hasattr(leafbio, 'emis') else 0.98

        # Shaded leaves
        denom_h = (rhoa * cp) / rac + rhoa * lambdah * e_to_q * sh / (rac + bch_rcw) + 4 * emis_leaf * sigmaSB * (Tch + 273.15) ** 3
        Tch = Tch + Wc * EBerch / denom_h

        # Sunlit leaves
        denom_u = (rhoa * cp) / rac + rhoa * lambdau * e_to_q * su / (rac + bcu_rcw) + 4 * emis_leaf * sigmaSB * (Tcu + 273.15) ** 3
        Tcu = Tcu + Wc * EBercu / denom_u

        # Soil
        rs_thermal = soil.rs_thermal if hasattr(soil, 'rs_thermal') else 0.05
        denom_s = rhoa * cp / ras + rhoa * lambdas * e_to_q * ss / (ras + rss) + 4 * (1 - rs_thermal) * sigmaSB * (Ts + 273.15) ** 3 + dG
        Ts = Ts + Wc * EBers / denom_s

        # Clamp unreasonable temperatures
        Tch = np.where(np.abs(Tch) > 100, Ta, Tch)
        Tcu = np.where(np.abs(Tcu) > 100, Ta, Tcu)

    # Print warnings
    if counter >= maxit:
        print('WARNING: maximum number of iterations exceeded')
        print(f'Maximum energy balance error sunlit vegetation = {maxEBercu:.2f} W m-2')
        print(f'Maximum energy balance error shaded vegetation = {maxEBerch:.2f} W m-2')
        print(f'Energy balance error soil = {maxEBers:.2f} W m-2')

    # Prepare outputs
    thermal_out = {
        'Tcu': Tcu,
        'Tch': Tch,
        'Tsu': Ts[1],
        'Tsh': Ts[0],
    }

    fluxes = {
        'Rnctot': aggregator(LAI, Rnuc, Rnhc, Fc, canopy, integr),
        'lEctot': aggregator(LAI, lEcu, lEch, Fc, canopy, integr),
        'Hctot': aggregator(LAI, Hcu, Hch, Fc, canopy, integr),
        'Actot': aggregator(LAI, bcu_A, bch_A, Fc, canopy, integr),
        'Tcave': aggregator(1, Tcu, Tch, Fc, canopy, integr),
        'Rnstot': np.dot(Fs_soil, Rns),
        'lEstot': np.dot(Fs_soil, lEs),
        'Hstot': np.dot(Fs_soil, Hs),
        'Gtot': np.dot(Fs_soil, G),
        'Tsave': np.dot(Fs_soil, Ts),
    }
    fluxes['Rntot'] = fluxes['Rnctot'] + fluxes['Rnstot']
    fluxes['lEtot'] = fluxes['lEctot'] + fluxes['lEstot']
    fluxes['Htot'] = fluxes['Hctot'] + fluxes['Hstot']

    # Biochemistry outputs
    bcu_out = type('obj', (object,), {
        'A': bcu_A,
        'Ag': bcu_Ag,
        'rcw': bcu_rcw,
        'Ci': bcu_Ci,
        'Ja': bcu_Ja,
        'eta': bcu_eta,
        'Kn': bcu_Kn,
    })()

    bch_out = type('obj', (object,), {
        'A': bch_A,
        'Ag': bch_Ag,
        'rcw': bch_rcw,
        'Ci': bch_Ci,
        'Ja': bch_Ja,
        'eta': bch_eta,
        'Kn': bch_Kn,
    })()

    resist_out.rss = rss

    return counter, rad, thermal_out, soil, bcu_out, bch_out, fluxes, resist_out, meteo
