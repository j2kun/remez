"""An implementation of the Remez algorithm."""

import math


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
  return [
      math.sin(math.pi * j / (2 * (n - 1))) for j in range(-n + 1, n - 1, 2)
  ]


def build_linear_system(cheb_coeffs, xs, fxs):
  """Builds the linear system solved per iteration of the Remez algorithm.

  Builds the linear system

  [b_0(x_0)  b_1(x_1)
  [
   ...
  [
  [
  """
