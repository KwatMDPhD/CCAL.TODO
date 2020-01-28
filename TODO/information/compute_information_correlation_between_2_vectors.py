from numpy import asarray, exp, log, nan, sign, sqrt, unique
from scipy.stats import pearsonr

from .estimate_joint_pdf import estimate_joint_pdf
from .unmesh import unmesh


def compute_information_correlation_between_2_vectors(
    vector_0, vector_1, fraction_grid_extension=0.1, n_grid=24,
):

    if unique(vector_0).size == 1 or unique(vector_1).size == 1:

        return nan

    vector_0 = (vector_0 - vector_0.mean()) / vector_0.std()

    vector_1 = (vector_1 - vector_1.mean()) / vector_1.std()

    r = pearsonr(vector_0, vector_1)[0]

    (x_grid, y_grid), pxy = unmesh(
        *estimate_joint_pdf(
            asarray((vector_0, vector_1)).T,
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
