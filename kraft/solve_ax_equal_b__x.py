from numpy import dot, full, nan
from numpy.linalg import pinv
from scipy.optimize import nnls


def solve_ax_equal_b__x(a, b, method):

    if method == "pinv":

        x = dot(pinv(a), b)

    elif method == "nnls":

        x = full((a.shape[1], b.shape[1]), nan)

        for i in range(b.shape[1]):

            x[:, i] = nnls(a, b[:, i])[0]

    return x
