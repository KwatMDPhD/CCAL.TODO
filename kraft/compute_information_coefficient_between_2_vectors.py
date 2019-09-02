from numpy import asarray, exp, log, nan, sign, sqrt, unique
from scipy.stats import pearsonr

from .compute_joint_probability import compute_joint_probability
from .FRACTION_GRID_EXTENSION import FRACTION_GRID_EXTENSION
from .N_GRID import N_GRID
from .unmesh import unmesh


def compute_information_coefficient_between_2_vectors(
    vector_0, y, fraction_grid_extension=FRACTION_GRID_EXTENSION, n_grid=N_GRID
):

    if unique(vector_0).size == 1 or unique(y).size == 1:

        return nan

    vector_0 = (vector_0 - vector_0.mean()) / vector_0.std()

    y = (y - y.mean()) / y.std()

    r = pearsonr(vector_0, y)[0]

    (x_grid, y_grid), pxy = unmesh(
        *compute_joint_probability(
            asarray((vector_0, y)).T,
            plot=False,
            dimension_bandwidth_factors=(1 - abs(r) * 2 / 3,) * 2,
            dimension_fraction_grid_extensions=(fraction_grid_extension,) * 2,
            dimension_n_grids=(n_grid,) * 2,
        )
    )

    dx = x_grid[1] - x_grid[0]

    dy = y_grid[1] - y_grid[0]

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    pxpy = asarray((px,) * n_grid).T * asarray((py,) * n_grid)

    mi = (pxy * log(pxy / pxpy)).sum() * dx * dy

    ic = sqrt(1 - exp(-2 * mi))

    return sign(r) * ic
