from numpy import dot, full, nan
from numpy.linalg import pinv
from pandas import DataFrame
from scipy.optimize import nnls


def solve_ax_b(a, b, method):

    a_ = a.values

    b_ = b.values

    if method == "pinv":

        x = dot(pinv(a_), b_)

    elif method == "nnls":

        x = full((a.shape[1], b.shape[1]), nan)

        for i in range(b.shape[1]):

            x[:, i] = nnls(a_, b_[:, i])[0]

    return DataFrame(x, index=a.columns, columns=b.columns)
