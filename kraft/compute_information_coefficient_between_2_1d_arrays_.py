from fastkde.fastKDE import pdf
from numpy import exp, finfo, full, log, nan, sign, sqrt, unique
from scipy.stats import pearsonr

eps = finfo(float).eps


def compute_information_coefficient_between_2_1d_arrays_(
    _1d_array_0, _1d_array_1, print_=False
):

    if unique(_1d_array_0).size == 1 or unique(_1d_array_1).size == 1:

        return nan

    fxy, (x_axis, y_axis) = pdf(_1d_array_0, _1d_array_1)

    fxy += eps
    if print_:
        print(f"F(x,y) shape = {fxy.shape}")
        print(f"F(x,y) min = {fxy.min()}")
        print(f"F(x,y) max = {fxy.max()}")
        print(f"F(x,y) sum = {fxy.sum()}")

    dx = x_axis[1] - x_axis[0]
    if print_:
        print(f"dx = {dx}")

    dy = y_axis[1] - y_axis[0]
    if print_:
        print(f"dy = {dy}")

    pxy = fxy / fxy.sum()
    if print_:
        print(f"P(x,y) min = {pxy.min()}")
        print(f"P(x,y) max = {pxy.max()}")
        print(f"P(x,y) sum = {pxy.sum()}")

    px = pxy.sum(axis=1) * dy
    if print_:
        print(f"P(x) min = {px.min()}")
        print(f"P(x) max = {px.max()}")
        print(f"P(x) sum = {px.sum()}")

    py = pxy.sum(axis=0) * dx
    if print_:
        print(f"P(y) min = {py.min()}")
        print(f"P(y) max = {py.max()}")
        print(f"P(y) sum = {py.sum()}")

    pxpy = full(py.size, px).T * full(px.size, py)
    if print_:
        print(f"P(x)P(y) min = {pxpy.min()}")
        print(f"P(x)P(y) max = {pxpy.max()}")
        print(f"P(x)P(y) sum = {pxpy.sum()}")

    mi = (pxy * log(pxy / pxpy)).sum() * dx * dy
    if print_:
        print(f"MI = {mi}")

    ic = sqrt(1 - exp(-2 * mi))
    if print_:
        print(f"IC = {ic}")

    # hxy = - (pxy * log(pxy)).sum() * dx * dy

    # hx = -(px * log(px)).sum() * dx

    # hy = -(py * log(py)).sum() * dy

    # mi = hx + hy - hxy

    return sign(pearsonr(_1d_array_0, _1d_array_1)[0]) * ic
