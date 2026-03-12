"""Leaf property averaging functions for SCOPE model.

Translated from: src/supporting/meanleaf.m

Functions for integrating leaf properties over leaf angles and canopy layers,
weighted by leaf inclination distribution and sunlit fractions.
"""

from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray


def meanleaf(
    F: NDArray[np.float64],
    lidf: NDArray[np.float64],
    nlazi: int,
    choice: Literal["angles", "layers", "angles_and_layers"],
    Ps: NDArray[np.float64] = None,
) -> NDArray[np.float64]:
    """Calculate weighted average of leaf properties.

    Integrates leaf properties over leaf angles and/or canopy layers,
    weighted by the leaf inclination distribution function (LIDF) and
    optionally by the sunlit fraction per layer.

    Args:
        F: Input array of leaf properties. Shape depends on choice:
           - 'angles': (nli, nlazi, nl) -> output (nl,)
           - 'layers': (nl,) -> output scalar
           - 'angles_and_layers': (nli, nlazi, nl) -> output scalar
           where nli=13 (inclination classes), nlazi=36 (azimuth classes),
           nl=number of layers.
        lidf: Leaf inclination distribution function. Shape (nli,).
        nlazi: Number of leaf azimuth classes (typically 36).
        choice: Integration method:
           - 'angles': Integrate over leaf angles only
           - 'layers': Integrate over layers only (requires Ps)
           - 'angles_and_layers': Integrate over both angles and layers
        Ps: Sunlit fraction per layer. Shape (nl,). Required for 'layers'
            and 'angles_and_layers' choices.

    Returns:
        Integrated values. Shape depends on choice:
        - 'angles': (nl,)
        - 'layers': scalar
        - 'angles_and_layers': scalar

    Example:
        >>> # Average photosynthesis over leaf angles for each layer
        >>> A_layer = meanleaf(A_per_angle, lidf, 36, 'angles')
        >>>
        >>> # Total canopy photosynthesis weighted by sunlit fraction
        >>> A_canopy = meanleaf(A_layer, lidf, 36, 'layers', Ps)
    """
    F = np.asarray(F, dtype=np.float64)
    lidf = np.asarray(lidf, dtype=np.float64)

    if choice == "angles":
        # Integration over leaf angles only
        # F shape: (nli, nlazi, nl)
        nli = F.shape[0]
        nl = F.shape[2]

        # Weight by LIDF and sum over inclination classes
        Fout = np.zeros_like(F)
        for j in range(nli):
            Fout[j, :, :] = F[j, :, :] * lidf[j]

        # Sum over inclination and azimuth, divide by nlazi
        Fout = np.sum(np.sum(Fout, axis=0), axis=0) / nlazi

        return Fout  # shape (nl,)

    elif choice == "layers":
        # Integration over layers only
        # F shape: (nl,) and Ps shape: (nl,)
        if Ps is None:
            raise ValueError("Ps (sunlit fraction) required for 'layers' integration")

        nl = len(F)
        return np.dot(Ps, F) / nl

    elif choice == "angles_and_layers":
        # Integration over both angles and layers
        # F shape: (nli, nlazi, nl)
        if Ps is None:
            raise ValueError("Ps (sunlit fraction) required for 'angles_and_layers' integration")

        nli = F.shape[0]
        nl = F.shape[2]

        # Weight by LIDF
        Fout = np.zeros_like(F)
        for j in range(nli):
            Fout[j, :, :] = F[j, :, :] * lidf[j]

        # Weight by sunlit fraction per layer
        for k in range(nl):
            Fout[:, :, k] = Fout[:, :, k] * Ps[k]

        # Sum over all dimensions
        return np.sum(Fout) / nlazi / nl

    else:
        raise ValueError(f"Invalid choice: {choice}. Use 'angles', 'layers', or 'angles_and_layers'")


def weighted_layer_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Calculate weighted mean over canopy layers.

    Args:
        values: Values per layer. Shape (nl,).
        weights: Weights per layer (e.g., LAI fraction). Shape (nl,).

    Returns:
        Weighted mean value.
    """
    return np.sum(values * weights) / np.sum(weights)


def sunlit_shaded_average(
    sunlit_values: NDArray[np.float64],
    shaded_values: NDArray[np.float64],
    sunlit_fraction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate combined sunlit/shaded average per layer.

    Args:
        sunlit_values: Values for sunlit leaves. Shape (nl,).
        shaded_values: Values for shaded leaves. Shape (nl,).
        sunlit_fraction: Fraction of sunlit leaves per layer. Shape (nl,).

    Returns:
        Combined average values per layer. Shape (nl,).
    """
    shaded_fraction = 1.0 - sunlit_fraction
    return sunlit_fraction * sunlit_values + shaded_fraction * shaded_values
