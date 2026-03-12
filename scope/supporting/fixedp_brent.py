"""Brent's method for fixed-point iteration.

Translated from: src/supporting/fixedp_brent_ari.m

This module implements Brent's method for finding fixed points of a function,
i.e., finding x such that f(x) = x, which is equivalent to finding the root
of g(x) = f(x) - x = 0.

The algorithm combines bisection, secant, and inverse quadratic interpolation
for robust and efficient convergence. This implementation follows the same
approach as brentq_numba in biochemical_numba.py.

Reference:
    Brent, R. P. (1971). An algorithm with guaranteed convergence for finding
    a zero of a function. The Computer Journal, 14(4), 422-425.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple, Union

ArrayLike = Union[float, NDArray[np.float64]]


def fixedp_brent(
    func: Callable[[ArrayLike], ArrayLike],
    x0: ArrayLike,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> Tuple[ArrayLike, ArrayLike, int]:
    """Find fixed point of func(x) using Brent's method.

    Finds x such that func(x) = x (equivalently, func(x) - x = 0).

    This implementation matches MATLAB's fixedp_brent_ari.m and is consistent
    with brentq_numba in biochemical_numba.py.

    The approach:
    1. Start from initial guess x0
    2. Compute b = func(x0) as second point
    3. If not bracketed, try secant extrapolation, then walking out
    4. Once bracketed, use standard Brent's method

    Args:
        func: Function to find fixed point of. Should return same shape as input.
        x0: Initial guess (scalar or array)
        tol: Tolerance for convergence (default 1e-7, matching MATLAB)
        max_iter: Maximum iterations (default 100)

    Returns:
        Tuple of (x_fixed, error, iterations):
            x_fixed: The fixed point solution
            error: Final error (func(x) - x)
            iterations: Number of function calls

    Example:
        >>> def f(x): return 0.5 * x + 1  # Fixed point at x=2
        >>> x, err, iters = fixedp_brent(f, 0.0)
        >>> print(f"x = {x:.6f}")  # x = 2.000000
    """
    # Handle scalar vs array input
    x0_arr = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    is_scalar = x0_arr.size == 1

    if is_scalar:
        result, err, fcounter = _fixedp_brent_scalar(func, x0_arr[0], tol, max_iter)
        return result, err, fcounter
    else:
        # Vectorized version for arrays
        return _fixedp_brent_vector(func, x0_arr, tol, max_iter)


def _fixedp_brent_scalar(
    func: Callable[[float], float],
    x0: float,
    tol: float,
    max_iter: int,
) -> Tuple[float, float, int]:
    """Scalar Brent's fixed-point method.

    Matches the logic from brentq_numba in biochemical_numba.py.
    """
    eps = 2.220446049250313e-16  # machine epsilon

    # MATLAB approach: start from x0, compute first iteration
    # a = x0 (initial guess)
    # b = func(a) (next iteration point)
    a = x0
    b = func(a)  # Ci_next(a) in MATLAB
    err_a = b - a  # err1 in MATLAB: f(a) - a
    err_b = func(b) - b  # err2 in MATLAB: f(b) - b
    fcounter = 2

    # Check if already converged
    if abs(err_b) < tol:
        return b, err_b, fcounter

    # Check if root is bracketed (signs differ)
    if err_a * err_b < 0:
        # Root is bracketed, proceed with Brent
        fa, fb = err_a, err_b
    else:
        # Root not bracketed - follow MATLAB's approach to find bracket

        # Step 1: Try secant extrapolation (MATLAB lines 66-81)
        if abs(err_b - err_a) > 1e-15:
            x1 = b - err_b * (b - a) / (err_b - err_a)
            # Clamp to reasonable range (for Ci, stay positive and not too large)
            x1 = max(0.0, min(2.0 * abs(x0), x1)) if x0 != 0 else max(0.0, x1)
            err_x1 = func(x1) - x1
            fcounter += 1

            # Check if x1 gives opposite sign
            if err_x1 * err_a < 0:
                # Found bracket with x1
                if abs(err_b) < abs(err_a):
                    a, err_a = b, err_b
                b, err_b = x1, err_x1
            elif err_x1 * err_b < 0:
                a, err_a = b, err_b
                b, err_b = x1, err_x1

        # Step 2: If still not bracketed, try walking out (MATLAB lines 93-116)
        if err_a * err_b > 0:
            # Make 'a' the point with smaller error magnitude
            if abs(err_b) < abs(err_a):
                a, b = b, a
                err_a, err_b = err_b, err_a

            # Both positive: walk a away from b (MATLAB lines 93-116)
            if err_a > 0 and err_b > 0:
                for _ in range(10):
                    diff_ab = b - a
                    a = a - diff_ab  # Double the distance from b
                    a = max(0.0, a)  # Don't go below 0
                    err_a = func(a) - a
                    fcounter += 1
                    if err_a * err_b < 0:
                        break  # Found bracket
                    if a <= 0:
                        break  # Can't go lower

            # Both negative: try x = 0 (MATLAB lines 119-127)
            # For Ci, A(0) gives Ci >= 0, so f(0) - 0 >= 0
            elif err_a < 0 and err_b < 0:
                a = 0.0
                err_a = func(a) - a
                fcounter += 1

        # Step 3: If still not bracketed, use damped fixed-point iteration
        if err_a * err_b > 0:
            # Fallback: damped fixed-point iteration
            Ci = x0
            damping = 0.5
            for _ in range(50):
                Ci_old = Ci
                Ci_new = func(Ci)
                Ci = damping * Ci_new + (1 - damping) * Ci_old
                fcounter += 1
                if abs(Ci - Ci_old) < tol:
                    break
            return Ci, func(Ci) - Ci, fcounter

        fa, fb = err_a, err_b

    # Now we have a bracket [a, b] with fa * fb < 0
    # Ensure |fb| <= |fa| (b is the better estimate)
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for i in range(max_iter):
        if abs(fb) < tol:
            return b, fb, fcounter

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol1 = 2 * eps * abs(b) + tol / 2
        m = (c - b) / 2

        if abs(m) <= tol1 or fb == 0:
            return b, fb, fcounter

        # Decide bisection or interpolation
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Linear interpolation (secant)
                p = 2 * m * s
                q = 1 - s
            else:
                # Inverse quadratic interpolation
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

        fb = func(b) - b
        fcounter += 1

        if fb * fc > 0:
            c = a
            fc = fa
            d = b - a
            e = d

    return b, fb, fcounter


def _fixedp_brent_vector(
    func: Callable[[NDArray], NDArray],
    x0: NDArray[np.float64],
    tol: float,
    max_iter: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Vectorized Brent's fixed-point method for array inputs.

    Applies the scalar algorithm element-wise with proper broadcasting.
    """
    n = x0.size
    result = np.zeros_like(x0)
    errors = np.zeros_like(x0)
    total_fcalls = 0

    # Create wrapper that handles single elements
    for i in range(n):
        def func_i(x):
            # Create full array, set element i, call func, return element i
            arr = x0.copy()
            arr[i] = x
            return func(arr)[i]

        result[i], errors[i], fcalls = _fixedp_brent_scalar(func_i, x0[i], tol, max_iter)
        total_fcalls += fcalls

    return result, errors, total_fcalls
