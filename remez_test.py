import math
import numpy as np
from numpy.testing import assert_allclose
import remez

cos_pi_4 = math.cos(math.pi / 4)
cos_pi_5 = math.cos(math.pi / 5)
cos_2pi_5 = math.cos(2 * math.pi / 5)
cos_3pi_5 = math.cos(3 * math.pi / 5)


def test_initial_reference_2():
  actual = remez.initial_reference(n=2)
  assert len(actual) == 2
  assert_allclose([-1, 1], actual)


def test_initial_reference_5():
  actual = remez.initial_reference(n=5)
  assert_allclose([-1, -cos_pi_4, 0, cos_pi_4, 1], actual)


def test_initial_reference_6():
  actual = remez.initial_reference(n=6)
  assert_allclose([-1, -cos_pi_5, -cos_2pi_5, cos_2pi_5, cos_pi_5, 1], actual)


def test_build_linear_system():
  xs = remez.initial_reference(n=5)
  fxs = np.sin(xs)
  A, b = remez.build_linear_system(xs, fxs)

  assert_allclose(b, fxs)
  expected_A = np.array([
      [1, -2, 3, -4, 1],  # x = -1
      [1, -math.sqrt(2), 1, 0, -1],  # x = -sqrt(2)/2
      [1, 0, -1, 0, 1],  # x = 0
      [1, math.sqrt(2), 1, 0, -1],  # x = sqrt(2)/2
      [1, 2, 3, 4, 1],  # x = 1
  ])
  soln = np.linalg.solve(A, b)
  assert remez.estimate_error(soln, np.sin) < 1e-02
