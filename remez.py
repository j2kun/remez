"""An implementation of the Remez algorithm."""

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
