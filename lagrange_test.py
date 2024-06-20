import lagrange
import remez


def test_lagrange_interpolation():
    # f(x) = x^2 + 3x + 2
    def f(x):
        return x * x + 3 * x + 2

    xs = remez.initial_reference(3)
    ys = [f(x) for x in xs]

    actual = lagrange.barycentric_lagrange(xs, ys)
    for (x, y) in zip(xs, ys):
        assert abs(f(x) - actual(x)) < 1e-08

    for i in range(-50, 50):
        assert abs(f(i) - actual(i)) < 1e-08

