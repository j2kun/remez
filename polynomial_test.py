import polynomial
import pytest


@pytest.mark.parametrize("count", list(range(6)))
def test_generate_chebyshev_polynomials(count):
  expected = [
      [1],
      [0, 2],
      [-1, 0, 4],
      [0, -4, 0, 8],
      [1, 0, -12, 0, 16],
  ][:count]
  actual = polynomial.generate_chebyshev_polynomials(count=count)
  assert expected == actual


def test_poly_eval():
  poly = [1, 2, 3, 4]
  x = 3
  expected = 1 + 2 * x + 3 * x * x + 4 * x * x * x
  assert polynomial.poly_eval(poly, x) == expected


def test_clenshaw_eval():
  poly = [1, 2, 3, 4]
  x = 3
  expected = polynomial.poly_eval(poly, x)
  actual = polynomial.clenshaw_eval(
      poly,
      x,
      # basis_0 = 1
      [1],
      # basis_1 = x
      [0, 1],
      # alpha(x) = x
      [0, 1],
      # beta(x) = 0
      [0],
  )
  assert expected == actual


def test_cheby_eval():
  fns = polynomial.generate_chebyshev_polynomials(count=3)
  coefficients = [1, 2, 3]
  x = 4
  expected = polynomial.poly_eval(
      polynomial.linear_combination(fns, coefficients), 4
  )
  actual = polynomial.cheb_eval(coefficients, x)
  assert expected == actual
