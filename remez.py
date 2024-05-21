"""An implementation of the Remez algorithm."""

import functools
import math
import numpy as np
import polynomial
import scipy


def initial_reference(n):
    """Compute n initial reference points as Chebyshev points

    The output points are ordered left to right on the interval [-1, 1].
    """
    if n == 0:
        return []
    if n == 1:
        return [0]

    """
  The values are most simply described as

      cos(pi * j / (n-1)) for 0 <= j <= n-1.

  But to enforce symmetry around the origin---broken by slight numerical
  inaccuracies---and the left-to-right ordering, we apply the identity

      cos(x + pi) = -cos(x) = sin(x - pi/2)

  to arrive at

      sin(pi*j/(n-1) - pi/2) = sin(pi * (2j - (n-1)) / (2(n-1)))

  An this is equivalent to the formula below, where the range of j is shifted
  and rescaled from {0, ..., n-1} to {-n+1, -n+3, ..., n-3, n-1}.

  For reference, cf. chebfun's chebpts.m
  https://github.com/chebfun/chebfun/blob/db207bc9f48278ca4def15bf90591bfa44d0801d/%40chebtech2/chebpts.m#L34
  """
    return [math.sin(math.pi * j / (2 * (n - 1))) for j in range(-n + 1, n, 2)]


def build_linear_system(xs, fxs):
    """Builds the linear system solved per iteration of the Remez algorithm.

    Builds the following linear system, where b_i is the i-th Chebyshev basis
    polynomial, sigma_i are the alternating signs for equioscillation, h is the
    leveled error, and f(x_i) are the values of f at each x value.

    [ b_0(x_0)   b_1(x_0)   ... b_n(x_0)   -sigma_0   ] [ c_0 ]   [ f(x_0)   ]
    [ b_0(x_1)   b_1(x_1)   ... b_n(x_1)   -sigma_1   ] [ c_1 ]   [ f(x_1)   ]
      ...        ...        ...            ...            ...   = [  ...     ]
    [ b_0(x_n)   b_1(x_n)   ... b_n(x_n)   -sigma_n   ] [ c_n ]   [ f(x_n)   ]
    [ b_0(x_n+1) b_1(x_n+1) ... b_n(x_n+1) -sigma_n+1 ] [ h   ]   [ f(x_n+1) ]

    Args:
      xs: the x_i used to evaluate the chebyshev polynomials T_i(x) and f(x).
      fxs: the values f(x_i).

    Returns:
      The values (A, b) forming the linear system above.
    """
    n = len(xs) - 2
    if n <= 0:
        raise ValueError("Need at least 3 points")

    if len(xs) != len(fxs):
        raise ValueError(
            f"Mismatching input lengths for xs ({len(xs)}) and fxs ({len(fxs)})"
        )

    # Writing as n+2 to emphasize that Remez needs n+2 points for a polynomial of
    # degree n.
    A = np.zeros((n + 2, n + 2), dtype=np.float32)
    b = np.zeros((n + 2,), dtype=np.float32)

    for j in range(n + 1):
        coeffs = np.zeros((j + 1,))
        coeffs[j] = 1
        A[:, j] = [polynomial.cheb_eval(coeffs, x) for x in xs]

    A[:, n + 1] = -((-1) ** np.arange(n + 2))
    return A, np.array(fxs)


def bisection_method(f, a, b, iters=100, precision=1e-15):
    """Apply the bisection method to find the roots of f within [a,b]."""
    fa = f(a)
    fb = f(b)

    if fa * fb >= 0:
        raise ValueError("invalid input to bisection method")

    for _ in range(iters):
        mid = (a + b) / 2
        fmid = f(mid)
        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid

        if abs(fmid) < precision or abs(b - a) < precision:
            return mid

    print(f"Warning: hit {iters} iters without reaching precision {precision}.")
    return mid


# def _secant_method_without_fallback(f, a, b, iters=100, precision=1e-15):
#     """Apply the secant method to find the roots of f within [a,b]."""
#     fa = f(a)
#     fb = f(b)
#
#     if fa * fb >= 0:
#         raise ValueError("invalid input to secant method")
#
#     xs = (a, b)
#     for _ in range(iters):
#         next_x = xs[-1] - f(xs[-1]) * (xs[-1] - xs[-2]) / (f(xs[-1]) - f(xs[-2]))
#         xs = (xs[-1], next_x)
#         f_next_x = f(next_x)
#         if abs(f_next_x) < precision or abs(xs[0] - xs[1]) < precision:
#             return next_x
#
#     print(f"Warning: hit {iters} iters without reaching precision {precision}.")
#     return next_x


def secant_method(f, a, b, iters=100):
    # result = _secant_method_without_fallback(f, a, b, iters)
    # if a <= result <= b:
    #     return result
    # fall back to bisection method
    return bisection_method(f, a, b, iters=iters)


def minimize(f, a, b, precision=1e-15):
    """Find the min of f within [a,b].

    Start with a golden section method.

    Later consider Brent's method, porting the scipy implementation at
    https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/optimize/_optimize.py#L2430
    Note that Brent's method requires golden section method as a subroutine,
    and otherwise uses quadratic interpolation to shrink the interval.
    """
    golden_ratio = (math.sqrt(5) - 1) / 2
    gr_times_b_minus_a = golden_ratio * (b - a)

    # x1 = a + (1 - golden_ratio)(b-a)
    x1 = b - gr_times_b_minus_a
    # x2 = a + golden_ratio(b-a)
    x2 = a + gr_times_b_minus_a
    f1 = f(x1)
    f2 = f(x2)

    while abs(b - a) > precision:
        if f1 > f2:
            # [a, b] -> [x1, b]
            a, x1, f1 = x1, x2, f2
            x2 = a + golden_ratio * (b - a)
            f2 = f(x2)
        else:
            # [a, b] -> [a, x2]
            b, x2, f2 = x2, x1, f1
            x1 = b - golden_ratio * (b - a)
            f1 = f(x1)

    return (a + b) / 2


def maximize(f, a, b):
    """Find the max of f within [a,b]."""
    return minimize(lambda x: -f(x), a, b)


def scipy_minimize(f, a, b):
    """Find the min of f within [a,b] using scipy."""
    result = scipy.optimize.minimize_scalar(f, bounds=(a, b), method="Bounded")
    if not result.success:
        raise ValueError(f"Failed to minimize {f} within [{a}, {b}]")
    return result.x


def dedupe_floats(xs, tol):
    """Remove duplicates from a list of floats."""
    seen = set()
    for x in xs:
        skip_x = False
        for y in seen:
            if abs(x - y) < tol:
                skip_x = True
                break

        if skip_x:
            continue
        seen.add(x)
    return list(sorted(list(seen)))


def find_roots(error_fn, xs, a, b):
    """Find n+1 roots of the error function, where len(xs) = n+2."""
    # Since f is arbitrary, the error function is also arbitrary.
    # In some cases we might be able to utilize a known derivative of f, and
    # compute a known derivative of p, and use that to apply Newton's method.
    # For this implementation we are arbitrary, and instead apply the secant
    # method.
    roots = []

    # The secant method will behave strangely when given two inputs that are
    # already close to existing roots. E.g., given roots r1, r2, r3, if you ask
    # it to solve for roots between [r1, r2] and [r2, r3] separately, it may
    # return r2 both times. To avoid this, we start by checking each point
    # to see if it's already a root or close to one, by bracketing around each
    # root and trying to solve for a root.
    interval_lengths = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    for i, x in enumerate(xs):
        err = abs(error_fn(x))
        if err < 1e-15:
            roots.append(x)
            continue

        if i == 0:
            delta = interval_lengths[i]
        elif i == len(xs) - 1:
            delta = interval_lengths[i - 1]
        else:
            delta = min(interval_lengths[i], interval_lengths[i - 1])

        left = max(a, x - delta / 2)
        right = min(b, x + delta / 2)
        if error_fn(left) * error_fn(right) < 0:
            roots.append(secant_method(error_fn, left, right))

    # Because the sign of the error oscillates by how the linear system was
    # constructed, there is a root between each successive pair of x_i.
    for i in range(len(xs) - 1):
        if error_fn(xs[i]) * error_fn(xs[i + 1]) < 0:
            roots.append(secant_method(error_fn, xs[i], xs[i + 1]))

    # Some roots may be duplicate, so dedupe
    roots = dedupe_floats(roots, 1e-07)
    if len(roots) < len(xs) - 1:
        raise ValueError(
            f"Expected n+1={len(xs) - 1} roots, but found {len(roots)}: {roots}"
        )
    return roots


def exchange(xs, error_fn, a, b):
    """Improve the reference point set for one iteration of the Remez algorithm.

    Args:
      xs: the current set of reference points x_i.
      error_fn: a function computing p(x_i) - f(x_i), where p is the current
        approximation and f is the desired function to approximate.
      a: the left endpoint of the interval.
      b: the right endpoint of the interval.

    Returns:
      The new reference point set.
    """
    # n+1 roots for n+2 reference points, and no duplicates
    roots = find_roots(error_fn, xs, a, b)

    # Now that we have the roots, there is a guaranteed extremum between each
    # pair of roots, including the endpoints of the interval [-1, 1]. Since
    # this set of extrema_bounds has size n+3, we end with n+2 new control
    # points.
    #
    # Note that find_roots may actually return n+2 roots, in situations where
    # the endpoints of the interval [a,b] happen to be roots. In this case, we
    # can't add the endpoints to bracket n+2 extrema. In fact, there may only
    # be n+1 extrema, not n+2. In that case, we need to add an arbitrary
    # additional point.
    extrema = []
    extrema_bounds = roots
    if roots[0] > a:
        extrema_bounds = [a] + extrema_bounds
    if roots[-1] < b:
        extrema_bounds = extrema_bounds + [b]

    for i in range(len(extrema_bounds) - 1):
        a, b = extrema_bounds[i], extrema_bounds[i + 1]
        # We could use the error function value at xs[i], but this may also
        # coincide with a root and mess up the flag to look for min/max.
        corresponding_error_value = error_fn((a + b) / 2)
        looking_for_min = corresponding_error_value < 0
        extrema.append(
            scipy_minimize(error_fn, a, b)
            if looking_for_min
            else scipy_minimize(lambda x: -error_fn(x), a, b)
        )

    new_points = extrema
    if len(new_points) == len(xs) - 1:
        # We need n+2 points, but only found n+1. Add an arbitrary point.
        new_points.append((xs[0] + extrema[0]) / 2)

    if len(new_points) < len(xs):
        raise ValueError(
            f"Expected {len(xs)} new points, but found {len(new_points)}: {new_points}"
        )

    return list(sorted(new_points))[: len(xs)]


def remez(f, n, max_iters=20):
    """Compute the the best degree-n polynomial approximation to f on [-1, 1].

    Returns the coefficients of the computed polynomial in the Chebyshev basis.
    """
    xs = initial_reference(n + 2)
    errs = []

    for iter in range(max_iters):
        fxs = [f(x) for x in xs]
        A, b = build_linear_system(xs, fxs)
        soln = np.linalg.solve(A, b)
        coeffs, leveled_error = soln[:-1], soln[-1]

        def error_fn(x):
            return polynomial.cheb_eval(coeffs, x) - f(x)

        error_fn = functools.lru_cache(maxsize=100)(error_fn)

        # validate the solution of the linear system
        for i, x in enumerate(xs):
            oscillating_error = (-1) ** i * leveled_error
            diff_from_expected = abs(error_fn(x) - oscillating_error)
            if diff_from_expected > 1e-06:
                raise ValueError(
                    f"Solution of linear system is not valid at x={x}; "
                    f"poly(x)={polynomial.cheb_eval(coeffs, x)}; "
                    f"f(x)={f(x)}; "
                    f"error={error_fn(x)}; "
                    f"expected_error={oscillating_error}; "
                    f"diff={error_fn(x) - oscillating_error}; "
                )

        # Anything less than 1e-07 and we get to the limit of the error
        # tolerance of the numpy linalg solver, which starts crapping out at
        # 1e-08. You can see this in action if you run this test with degree 10
        # and plot the first error function. The interpolation is so good that
        # the leveled error is off by up to 0.5e-08, at which point root
        # finding fails because the error is not truly oscillating.
        if max(abs(error_fn(x)) for x in xs) < 1e-08:
            break

        xs = exchange(xs, error_fn, -1, 1)
        plot_error(error_fn, -1, 1, xs, leveled_error, iter)
        errs.append(max(abs(error_fn(x)) for x in xs))

        print(f"{iter=}: {errs[-1]=}")
        if len(errs) > 1 and abs((errs[-1] - errs[-2]) / errs[-2]) < 1e-03:
            break

    return coeffs


def estimate_error(cheb_coeffs, f):
    xs = np.linspace(-1, 1, 100)
    pxs = np.array([polynomial.cheb_eval(cheb_coeffs, x) for x in xs])
    fxs = np.array([f(x) for x in xs])
    return np.max(pxs - fxs)


def plot_error(fn, a, b, ref_pts, leveled_error, i):
    import matplotlib.pyplot as plt

    # clear the plot from last time
    plt.clf()

    xs = np.linspace(a, b, 250)
    err = np.array([fn(x) for x in xs])
    plt.plot(xs, err, "g")

    # and plot the reference points from xs along the curve
    ys = [fn(x) for x in ref_pts]
    plt.scatter(ref_pts, ys, c="r")

    # and draw the x-axis
    plt.axhline(y=0, color="k")
    plt.axhline(y=leveled_error, color="k")
    plt.axhline(y=-leveled_error, color="k")

    plt.savefig(f"error_{i:03d}.pdf")
    # raise ValueError("wat")


if __name__ == "__main__":
    xs = initial_reference(100)
    fxs = np.sin(xs)
    A, b = build_linear_system(xs, fxs)
    soln = np.linalg.solve(A, b)

    # plot
    import matplotlib.pyplot as plt

    xs = np.linspace(-1, 1, 1000)
    pxs = np.array([polynomial.cheb_eval(soln, x) for x in xs])
    fxs = np.sin(xs)
    err = pxs - fxs

    # plt.plot(xs, pxs, "r")
    # plt.plot(xs, fxs, "b")
    plt.plot(xs, err, "g")
    plt.show()
