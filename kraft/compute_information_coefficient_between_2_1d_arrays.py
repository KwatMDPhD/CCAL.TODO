from fastkde.fastKDE import pdf
from numpy import full, exp, finfo, isnan, log, nan, sign, sqrt, unique
from scipy.stats import pearsonr

eps = finfo(float).eps


def compute_information_coefficient_between_2_1d_arrays(_1d_array_0, _1d_array_1):

    pearson_correlation = pearsonr(_1d_array_0, _1d_array_1)[0]

    if (
        isnan(pearson_correlation)
        or unique(_1d_array_0).size == 1
        or unique(_1d_array_1).size == 1
    ):

        return nan

    fxy, (x_axis, y_axis) = pdf(_1d_array_0, _1d_array_1, axisExpansionFactor=2)

    fxy += eps
    # print(f"F(x,y) min = {fxy.min()}")
    # print(f"F(x,y) max = {fxy.max()}")
    # print(f"F(x,y) sum = {fxy.sum()}")

    dx = x_axis[1] - x_axis[0]
    # print(f"dx = {dx}")

    dy = y_axis[1] - y_axis[0]
    # print(f"dy = {dy}")

    pxy = fxy / (fxy.sum() * dx * dy)
    # print(f"P(x,y) min = {pxy.min()}")
    # print(f"P(x,y) max = {pxy.max()}")
    # print(f"P(x,y) sum = {pxy.sum()}")

    px = pxy.sum(axis=1) * dy
    # print(f"P(x) min = {px.min()}")
    # print(f"P(x) max = {px.max()}")
    # print(f"P(x) sum = {px.sum()}")

    py = pxy.sum(axis=0) * dx
    # print(f"P(y) min = {py.min()}")
    # print(f"P(y) max = {py.max()}")
    # print(f"P(y) sum = {py.sum()}")

    pxpy = full(py.size, px).T * full(px.size, py)
    # print(f"P(x)P(y) min = {pxpy.min()}")
    # print(f"P(x)P(y) max = {pxpy.max()}")
    # print(f"P(x)P(y) sum = {pxpy.sum()}")

    mi = (pxy * log(pxy / pxpy)).sum() * dx * dy
    # print(f"MI = {mi}")

    # hxy = - (pxy * log(pxy)).sum() * dx * dy

    # hx = -(px * log(px)).sum() * dx

    # hy = -(py * log(py)).sum() * dy

    # mi = hx + hy - hxy

    return sign(pearson_correlation) * sqrt(1 - exp(-2 * mi))
