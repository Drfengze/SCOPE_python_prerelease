"""Numerical integration routines for SCOPE model.

Translated from: src/supporting/Sint.m
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def sint(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """Simpson/trapezoidal integration of y over x.

    Computes the integral of y with respect to x using the trapezoidal rule
    (midpoint approximation): sum of (y[i] + y[i+1])/2 * (x[i+1] - x[i])

    Note: Despite the name "Sint" (Simpson integration) in the original MATLAB,
    the implementation uses the trapezoidal rule.

    Args:
        y: Values to integrate. Can be 1D array of shape (n,) or 2D array of
           shape (m, n) where integration is performed along the last axis.
        x: Integration variable. Must be 1D array of shape (n,) and
           monotonically increasing.

    Returns:
        Integrated values. For 1D input y: scalar.
        For 2D input y with shape (m, n): array of shape (m,).

    Example:
        >>> x = np.linspace(0, 1, 101)
        >>> y = x ** 2
        >>> result = sint(y, x)  # Should be ~0.333
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Ensure x is 1D
    x = x.ravel()
    nx = len(x)

    # Calculate step sizes
    step = x[1:nx] - x[0 : nx - 1]  # shape (nx-1,)

    # Handle 1D and 2D cases
    if y.ndim == 1:
        # 1D case: simple trapezoidal integration
        mean = 0.5 * (y[0 : nx - 1] + y[1:nx])
        result = np.dot(mean, step)
    else:
        # 2D case: integrate along last axis
        # Ensure y has shape (m, n) where n matches len(x)
        if y.shape[-1] != nx:
            # Transpose if needed (MATLAB style where rows are spectra)
            if y.shape[0] == nx:
                y = y.T

        # Midpoint values: shape (m, nx-1)
        mean = 0.5 * (y[:, 0 : nx - 1] + y[:, 1:nx])

        # Integrate: (m, nx-1) @ (nx-1,) -> (m,)
        result = np.dot(mean, step)

    return result


def cumulative_integral(
    y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cumulative integration of y over x.

    Computes the cumulative integral from x[0] to each x[i].

    Args:
        y: Values to integrate. Shape (n,).
        x: Integration variable. Shape (n,), monotonically increasing.

    Returns:
        Cumulative integral values. Shape (n,), starting from 0.

    Example:
        >>> x = np.linspace(0, 1, 101)
        >>> y = np.ones(101)
        >>> result = cumulative_integral(y, x)  # Linear ramp from 0 to 1
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    nx = len(x)
    step = x[1:nx] - x[0 : nx - 1]
    mean = 0.5 * (y[0 : nx - 1] + y[1:nx])

    # Cumulative sum of step * mean
    cumsum = np.zeros(nx, dtype=np.float64)
    cumsum[1:] = np.cumsum(mean * step)

    return cumsum


def spectral_integral(
    spectrum: NDArray[np.float64],
    wavelength: NDArray[np.float64],
    wl_start: float = None,
    wl_end: float = None,
) -> float:
    """Integrate a spectrum over a wavelength range.

    Convenience function for integrating spectral quantities.

    Args:
        spectrum: Spectral values [per nm or per wavelength unit]
        wavelength: Wavelength array [nm]
        wl_start: Start wavelength for integration [nm]. Default: min(wavelength).
        wl_end: End wavelength for integration [nm]. Default: max(wavelength).

    Returns:
        Integrated value over the wavelength range.

    Example:
        >>> # Integrate irradiance spectrum from 400-700 nm for PAR
        >>> wl = np.arange(400, 701, 1)
        >>> irradiance = np.ones(301)  # 1 W/m²/nm
        >>> par = spectral_integral(irradiance, wl, 400, 700)  # 300 W/m²
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.float64)

    if wl_start is None:
        wl_start = wavelength.min()
    if wl_end is None:
        wl_end = wavelength.max()

    # Find indices within range
    mask = (wavelength >= wl_start) & (wavelength <= wl_end)
    wl_subset = wavelength[mask]
    spec_subset = spectrum[mask]

    return sint(spec_subset, wl_subset)
