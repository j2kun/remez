from chebyshev import generate_chebyshev_polynomials
import pytest


@pytest.mark.parametrize("count", list(range(6)))
def test_generate_chebyshev_polynomials(count):
  expected = [
      [1],
      [0, 1],
      [-1, 0, 2],
      [0, -3, 0, 4],
      [1, 0, -8, 0, 8],
  ][:count]
  actual = generate_chebyshev_polynomials(count=count)
  assert expected == actual
