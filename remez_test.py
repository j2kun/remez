import math
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
