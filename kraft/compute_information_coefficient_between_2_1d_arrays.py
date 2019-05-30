from numpy import asarray, exp, log, nan, sign, sqrt, unique
from scipy.stats import pearsonr

from .estimate_kernel_density import estimate_kernel_density
from .unmesh import unmesh


def compute_information_coefficient_between_2_1d_arrays(x, y, n_grid=64):

    if unique(x).size == 1 or unique(y).size == 1:

        return nan

    x = (x - x.mean()) / x.std()

    y = (y - y.mean()) / y.std()

    r = pearsonr(x, y)[0]

    (x_grid, y_grid), fxy = unmesh(
        *estimate_kernel_density(
            asarray((x, y)).T,
            dimension_bandwidth_factors=(1 - abs(r) * 2 / 3,) * 2,
            dimension_n_grids=(n_grid,) * 2,
            plot=False,
        )
    )

    dx = x_grid[1] - x_grid[0]

    dy = y_grid[1] - y_grid[0]

    pxy = fxy / (fxy.sum() * dx * dy)

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    pxpy = asarray((px,) * n_grid).T * asarray((py,) * n_grid)

    mi = (pxy * log(pxy / pxpy)).sum() * dx * dy

    ic = sqrt(1 - exp(-2 * mi))

    return sign(r) * ic
