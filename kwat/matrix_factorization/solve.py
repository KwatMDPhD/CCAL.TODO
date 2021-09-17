from numpy import dot, full, nan
from numpy.linalg import pinv
from pandas import DataFrame
from scipy.optimize import nnls


def solve(daa, dab, me):

    maa = daa.values

    mab = dab.values

    if me == "pinv":

        mas = dot(pinv(maa), mab)

    elif me == "nnls":

        mas = full([daa.shape[1], dab.shape[1]], nan)

        for ie in range(dab.shape[1]):

            mas[:, ie] = nnls(maa, mab[:, ie])[0]

    return DataFrame(data=mas, index=daa.columns, columns=dab.columns)
