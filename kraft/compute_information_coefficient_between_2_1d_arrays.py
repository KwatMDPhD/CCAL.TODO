from KDEpy import FFTKDE
from numpy import asarray, exp, finfo, linspace, log, nan, sign, sqrt, unique
from scipy.stats import pearsonr

from .ALMOST_ZERO import ALMOST_ZERO
from .compute_1d_array_bandwidth import compute_1d_array_bandwidth
from .make_mesh_grid_point_x_dimension import make_mesh_grid_point_x_dimension

eps = finfo(float).eps


def compute_information_coefficient_between_2_1d_arrays(x, y, n_grid=64):

    if unique(x).size == 1 or unique(y).size == 1:

        return nan

    x = (x - x.mean()) / x.std()

    y = (y - y.mean()) / y.std()

    r = pearsonr(x, y)[0]

    bandwidth_factor = 1 - abs(r) * 2 / 3

    x_bw = compute_1d_array_bandwidth(x) * bandwidth_factor

    y_bw = compute_1d_array_bandwidth(y) * bandwidth_factor

    x_grid = linspace(x.min() - 1, x.max() + 1, num=n_grid)

    y_grid = linspace(y.min() - 1, y.max() + 1, num=n_grid)

    fxy = (
        (
            FFTKDE(bw=(x_bw, y_bw))
            .fit(asarray((x, y)).T)
            .evaluate(make_mesh_grid_point_x_dimension((x_grid, y_grid)))
        )
        .reshape(n_grid, n_grid)
        .T
    ).clip(min=ALMOST_ZERO)

    dx = x_grid[1] - x_grid[0]

    dy = y_grid[1] - y_grid[0]

    pxy = fxy / (fxy.sum() * dx * dy)

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    pxpy = asarray((px,) * n_grid).T * asarray((py,) * n_grid)

    mi = (pxy * log(pxy / pxpy)).sum() * dx * dy

    # hxy = - (pxy * log(pxy)).sum() * dx * dy
    #
    # hx = -(px * log(px)).sum() * dx
    #
    # hy = -(py * log(py)).sum() * dy
    #
    # mi = hx + hy - hxy

    ic = sqrt(1 - exp(-2 * mi))

    # print("=" * 80)
    # print(f"R = {r}")
    # print(f"X BW = {x_bw:.3f}")
    # print(f"Y BW = {y_bw:.3f}")
    # print("-" * 80)
    # print(f"X Grid (size = {x_grid.size} & dx = {dx:.3f}) =")
    # print(x_grid)
    # print(f"Y Grid (size = {y_grid.size} & dy = {dy:.3f}) =")
    # print(y_grid)
    # print("-" * 80)
    # print("F(x,y) =")
    # print(fxy)
    # print(f"F(x,y) min = {fxy.min():.3e}")
    # print(f"F(x,y) max = {fxy.max():.3f}")
    # print("-" * 80)
    # print(f"P(x,y) min = {pxy.min():.3e}")
    # print(f"P(x,y) max = {pxy.max():.3f}")
    # print("-" * 80)
    # print(f"P(x) min = {px.min():.3e}")
    # print(f"P(x) max = {px.max():.3f}")
    # print("-" * 80)
    # print(f"P(y) min = {py.min():.3e}")
    # print(f"P(y) max = {py.max():.3f}")
    # print("-" * 80)
    # print(f"P(x)P(y) min = {pxpy.min():.3e}")
    # print(f"P(x)P(y) max = {pxpy.max():.3f}")
    # print("-" * 80)
    # print(f"MI = {mi:.3e}")
    # print(f"IC = {ic:.3f}")
    # print("^" * 80)

    return sign(r) * ic
