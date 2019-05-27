import rpy2.robjects as ro
from numpy import asarray, exp, finfo, isnan, log, nan, sign, sqrt, unique
from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr

eps = finfo(float).eps

ro.conversion.py2ri = numpy2ri

mass = importr("MASS")


def compute_information_coefficient_between_2_1d_arrays(
    _1d_array_0, _1d_array_1, n_grid=24
):

    pearson_correlation = pearsonr(_1d_array_0, _1d_array_1)[0]

    if (
        isnan(pearson_correlation)
        or unique(_1d_array_0).size == 1
        or unique(_1d_array_1).size == 1
    ):

        return nan

    dx = (_1d_array_0.max() - _1d_array_0.min()) / (n_grid - 1)
    print(f"dx = {dx}")

    dy = (_1d_array_1.max() - _1d_array_1.min()) / (n_grid - 1)
    print(f"dy = {dy}")

    pearson_correlation_abs = abs(pearson_correlation)

    bandwidth_x = mass.bcv(_1d_array_0)[0] * (1 - pearson_correlation_abs * 0.75)

    bandwidth_y = mass.bcv(_1d_array_1)[0] * (1 - pearson_correlation_abs * 0.75)

    fxy = asarray(
        mass.kde2d(
            _1d_array_0,
            _1d_array_1,
            asarray((bandwidth_x, bandwidth_y)),
            n=asarray((n_grid,)),
        )[2]
    )

    fxy += eps

    pxy = fxy / (fxy.sum() * dx * dy)
    print(f"P(x,y) min = {pxy.min()}")
    print(f"P(x,y) max = {pxy.max()}")
    print(f"P(x,y) sum = {pxy.sum()}")

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    mi = (
        (pxy * log(pxy / (asarray((px,) * n_grid).T * asarray((py,) * n_grid)))).sum()
        * dx
        * dy
    )

    # hxy = - (pxy * log(pxy)).sum() * dx * dy

    # hx = -(px * log(px)).sum() * dx

    # hy = -(py * log(py)).sum() * dy

    # mi = hx + hy - hxy

    return sign(pearson_correlation) * sqrt(1 - exp(-2 * mi))
