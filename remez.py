"""An implementation of the Remez algorithm."""

import functools
import math
import numpy as np
import polynomial


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
  levelled error, and f(x_i) are the values of f at each x value.

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

  # (-1)**j; does it matter if I flip all these signs?
  A[:, n + 1] = (-1) ** np.arange(n + 2)
  return A, np.array(fxs)


def secant_method(f, a, b, iters=100, precision=1e-15):
  """Apply the secant method to find the roots of f within [a,b]."""
  if f(a) * f(b) > 0:
    raise ValueError("f(a) and f(b) must have different signs.")

  xs = (a, b)
  for _ in range(iters):
    next_x = xs[-1] - f(xs[-1]) * (xs[-1] - xs[-2]) / (f(xs[-1]) - f(xs[-2]))
    xs = (xs[-1], next_x)
    f_next_x = f(next_x)
    if abs(f_next_x) < precision or abs(xs[0] - xs[1]) < precision:
      return next_x

  print(f"Warning: hit {iters} iters without reaching precision {precision}.")
  return next_x


def compute_min(f, a, b):
  """Find the min of f within [a,b].

  Consider Brent's method, porting the scipy implementation at
  https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/optimize/_optimize.py#L2430
  """
  pass


def compute_max(f, a, b):
  """Find the max of f within [a,b]."""
  return compute_min(lambda x: -f(x), a, b)


def exchange(xs, errs, levelled_error, cheb_coeffs, f):
  """Improve the reference point set for one iteration of the Remez algorithm.

  Args:
    xs: the current set of reference points x_i.
    errs: the values of p(x_i) - f(x_i), where p is the current approximation
      and f is the desired function to approximate.
    levelled_error: the value h for which p(x_i) = f(x_i) +/- h (n.b., with
      alternating signs as i increases) for all i.
    cheb_coeffs: the Chebyshev coefficients of the current candidate polynomial.
    f: the function to be approximated, as a callable.

  Returns:
    - The new reference point set.
    - The differences between the error function values at the old and new
      reference points.
  """
  # Step 1: find the roots of the error function.
  # Since f is arbitrary, the error function is also arbitrary.
  # In some cases we might be able to utilize a known derivative of f, and
  # compute a known derivative of p, and use that to apply Newton's method.
  # For this implementation we are arbitrary, and instead apply the secant
  # method.
  roots = []

  @functools.lru_cache(maxsize=100)
  def error_fn(x):
    return polynomial.cheb_eval(cheb_coeffs, x) - f(x)

  for i in range(len(xs) - 1):
    # Because the sign of the error oscillates, there is a root between
    # each successive pair of x_i.
    roots.append(secant_method(error_fn, xs[i], xs[i + 1]))

  # Now that we have the roots, there is a guaranteed extremum between
  # each pair of roots, including the endpoints of the interval [-1, 1].
  # Since this set of extrema_bounds has size n+3, we end with n+2 new
  # control points.
  extrema = []
  extrema_bounds = [-1] + roots + [1]
  for i in range(len(extrema_bounds) - 1):
    corresponding_error_value = errs[i]
    looking_for_min = corresponding_error_value < 0
    a, b = extrema_bounds[i], extrema_bounds[i + 1]
    extrema.append(
        compute_min(error_fn, a, b)
        if looking_for_min
        else compute_max(error_fn, a, b)
    )

  for ext in extrema:
    if error_fn(ext) < h:
      print(
          "Warning: found extrema that are worse than starting point."
          " Shenanigans are afoot!"
      )

  diffs = [error_fn(ext) - err for (ext, err) in zip(extrema, errs)]
  return (extrema, diffs)


def estimate_error(cheb_coeffs, f):
  xs = np.linspace(-1, 1, 100)
  pxs = np.array([polynomial.cheb_eval(cheb_coeffs, x) for x in xs])
  fxs = np.array([f(x) for x in xs])
  return np.max(pxs - fxs)


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
