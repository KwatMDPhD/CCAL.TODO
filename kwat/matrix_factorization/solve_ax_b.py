from numpy import dot, full, nan
from numpy.linalg import pinv
from pandas import DataFrame
from scipy.optimize import nnls


def solve_ax_b(daa, dab, me):

    am = daa.values

    bm = dab.values

    if me == "pinv":

        xm = dot(pinv(am), bm)

    elif me == "nnls":

        xm = full([daa.shape[1], dab.shape[1]], nan)

        for ie in range(dab.shape[1]):

            xm[:, ie] = nnls(am, bm[:, ie])[0]

    return DataFrame(data=xm, index=daa.columns, columns=dab.columns)
