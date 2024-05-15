"""Functions related to polynomials."""

from itertools import zip_longest


def linear_combination(
    polys: list[list[float]], weights: list[float]
) -> list[float]:
  assert len(weights) == len(polys)
  return [
      sum([c * x for (c, x) in zip(weights, coeffs)])
      for coeffs in zip_longest(*polys, fillvalue=0)
  ]


def generate_chebyshev_polynomials(count: int) -> list[list[float]]:
  """Generate chebyshev polynomials in the monomial basis.

  T_0(x) = 1
  T_1(x) = 2x
  T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)

  Returns a list of `count` polynomials T_0(x), ..., T_{count - 1}(x). Each list
  is a polynomial whose nonzero coefficients are listed in increasing order of
  degree.
  """
  if count < 1:
    return []

  polynomials = [[1], [0, 2]]
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


def poly_eval(poly: list[float], x: float) -> float:
  """Evaluate a polynomial in the monomial basis."""
  acc = 0
  for c in reversed(poly):
    acc = acc * x + c
  return acc


def clenshaw_eval(
    coefficients: list[float],
    value: float,
    basis_0: list[float],
    basis_1: list[float],
    alpha: list[float],
    beta: list[float],
) -> float:
  """Evaluate a linear combination of polynomial basis functions.

  The basis must adhere to the three-term recurrence:

    T_k(x) = alpha(x) T_{k-1}(x) + beta(x) T_{k-2}(x)

  Args:
    coefficients: the coefficients c_k of the function to evaluate, i.e., f(x) =
      sum_k c_k T_k(x)
    value: the input to f(x) to evaluate
    basis_0: the polynomial T_0(x)
    basis_1: the polynomial T_1(x)
    alpha: the coefficient alpha(x) of the recurrence
    beta: the coefficient beta(x) of the recurrence

  Returns:
    f(value)
  """
  # Since these are generally trivial (monomials or constants), one could
  # trivially speed this up for a specific recurrence by inlining the values and
  # avoiding the loop/list construction in poly_eval.
  alpha_x = poly_eval(alpha, value)
  beta_x = poly_eval(beta, value)
  basis_0_x = poly_eval(basis_0, value)
  basis_1_x = poly_eval(basis_1, value)

  # b[0] = b_{k+1} and b[1] = b_{k+2}
  b = (0, 0)

  # The constant term is skipped, incorporated at the end
  for c in reversed(coefficients[1:]):
    b_k = c + alpha_x * b[0] + beta_x * b[1]
    b = (b_k, b[0])

  return (
      basis_0_x * coefficients[0] + basis_1_x * b[0] + beta_x * basis_0_x * b[1]
  )


def cheb_eval(coefficients: list[float], x: float) -> float:
  """Evaluate a truncated Chebyshev series at x."""
  basis = generate_chebyshev_polynomials(count=2)
  alpha = [0, 2]  # alpha(x) = 2x
  beta = [-1]  # beta(x) = -1
  return clenshaw_eval(coefficients, x, basis[0], basis[1], alpha, beta)
