"""Routines for Lagrange interpolation."""


def compute_barycentric_weights(xs, a=-1, b=1):
    """Compute barycentric weights for Lagrange interpolation."""
    n = len(xs)
    weights = [1] * n
    for j in range(n):
        for k in range(n):
            if j == k:
                continue
            weights[j] *= xs[j] - xs[k]

    C = (b - a) / 4
    C_inv = 1 / C
    for j in range(n):
        weights[j] = 1.0 / (C_inv * weights[j])

    return weights


def compute_barycentric_weights_chebyshev(xs, a=-1, b=1):
    """Compute barycentric weights for Lagrange interpolation
    on Chebyshev nodes of the second kind in [-1, 1].

    Computes the formula w_j = (-1)^j d_j for j = 0, ..., n=len(xs),
    where d_0 = d_n = 0.5, d_j = 1 otherwise.
    """
    n = len(xs)
    weights = [1] * n
    weights[0] = 0.5
    weights[-1] = 0.5
    for j in range(1, n, 2):
        weights[j] = -weights[j]

    # This is the official formula for the weight change, but it's not
    # necessary because it occurs in the numerator and denominator of the
    # interpolation formula.
    #
    # if a != -1 or b != 1:
    #     scale = 2**n / (b - a)**n
    #     for j in range(n):
    #         weights[j] *= scale

    return weights


def barycentric_lagrange(xs, ys, weights_computer=compute_barycentric_weights):
    """Compute the Lagrange interpolating polynomial of the points (x, y)."""
    weights = weights_computer(xs)

    def f(x):
        numerator = 0
        denominator = 0
        for w, xi, yi in zip(weights, xs, ys):
            if x == xi:
                return yi
            common_factor = w / (x - xi)
            numerator += yi * common_factor
            denominator += common_factor
        return numerator / denominator

    return f
