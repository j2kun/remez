"""Functions related to Chebyshev polynomials."""

from itertools import zip_longest


def generate_chebyshev_polynomials(count: int) -> list[list[int]]:
  """Generate chebyshev polynomials in the monomial basis.

  T_0(x) = 1
  T_1(x) = x
  T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)

  Returns a list of `count` polynomials T_0(x), ..., T_{count - 1}(x). Each list
  is a polynomial whose nonzero coefficients are listed in increasing order of
  degree.
  """
  if count < 1:
    return []

  polynomials = [[1], [0, 1]]
  if count <= 2:
    return polynomials[:count]

  for i in range(2, count):
    x_k_minus_1 = [0] + polynomials[-1]
    k_minus_2 = polynomials[-2]
    next_polynomial = [
        2 * x - y for (x, y) in zip_longest(x_k_minus_1, k_minus_2, fillvalue=0)
    ]
    polynomials.append(next_polynomial)

  return polynomials
