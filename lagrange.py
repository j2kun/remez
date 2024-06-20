"""Routines for Lagrange interpolation."""

def compute_barycentric_weights(xs):
    """Compute barycentric weights for Lagrange interpolation."""
    n = len(xs)
    weights = [1] * n
    for j in range(n):
        for k in range(n):
            if j == k:
                continue
            weights[j] *= (xs[j] - xs[k])

    for j in range(n):
        weights[j] = 1.0 / weights[j]

    return weights


def barycentric_lagrange(xs, ys):
    """Compute the Lagrange interpolating polynomial of the points (x, y)."""
    weights = compute_barycentric_weights(xs)
    def f(x):
        numerator = 0
        denominator = 0
        for (w, xi, yi) in zip(weights, xs, ys):
            if x == xi:
                return yi
            common_factor = w / (x - xi)
            numerator += yi * common_factor
            denominator += common_factor
        return numerator / denominator

    return f

