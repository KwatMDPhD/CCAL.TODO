from numpy import asarray, exp, nan, outer, sign, sqrt, unique
from scipy.stats import pearsonr

from .estimate_pdf import estimate_pdf
from .get_bandwidth import get_bandwidth
from .get_kl import get_kl
from .make_grid import make_grid
from .normalize import normalize
from .unmesh import unmesh


def get_ic(vector_0, vector_1):

    if unique(vector_0).size == 1 or unique(vector_1).size == 1:

        return nan

    vector_0 = normalize(vector_0, "-0-")

    vector_1 = normalize(vector_1, "-0-")

    r = pearsonr(vector_0, vector_1)[0]

    bandwidth_factor = 1 - abs(r) * 2 / 3

    bandwidths = (
        get_bandwidth(vector_0) * bandwidth_factor,
        get_bandwidth(vector_1) * bandwidth_factor,
    )

    fraction_grid_extension = 0.1

    n_grid = 24

    grids = (
        make_grid(vector_0.min(), vector_0.max(), fraction_grid_extension, n_grid),
        make_grid(vector_1.min(), vector_1.max(), fraction_grid_extension, n_grid),
    )

    (x_grid, y_grid), pxy = unmesh(
        *estimate_pdf(
            asarray((vector_0, vector_1)).T,
            plot=False,
            bandwidths=bandwidths,
            grids=grids,
        )
    )

    dx = x_grid[1] - x_grid[0]

    dy = y_grid[1] - y_grid[0]

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    pxpy = outer(px, py)

    mi = get_kl(pxy, pxpy).sum() * dx * dy

    ic = sqrt(1 - exp(-2 * mi))

    return sign(r) * ic
