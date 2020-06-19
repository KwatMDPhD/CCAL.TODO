from numpy import asarray, exp, log, nan, outer, sign, sqrt, unique
from scipy.stats import pearsonr

from .array import normalize
from .kernel_density import get_bandwidth
from .point_x_dimension import get_grids, grid, reshape
from .probability import get_pdf


def get_entropy(vector):

    probability = vector / vector.sum()

    return -(probability * log(probability)).sum()


def get_kld(vector_0, vector_1):

    return vector_0 * log(vector_0 / vector_1)


def get_jsd(vector_0, vector_1, vector_reference=None):

    if vector_reference is None:

        vector_reference = (vector_0 + vector_1) / 2

    kld_0 = get_kld(vector_0, vector_reference)

    kld_1 = get_kld(vector_1, vector_reference)

    return kld_0, kld_1, kld_0 - kld_1


def get_zd(vector_0, vector_1):

    kld_0 = get_kld(vector_0, vector_1)

    kld_1 = get_kld(vector_1, vector_0)

    return kld_0, kld_1, kld_0 - kld_1


def get_ic(vector_0, vector_1):

    # TODO: consider error
    if unique(vector_0).size == 1 or unique(vector_1).size == 1:

        return nan

    vector_0 = normalize(vector_0, "-0-")

    vector_1 = normalize(vector_1, "-0-")

    r = pearsonr(vector_0, vector_1)[0]

    bandwidth_factor = 1 - abs(r) * 2 / 3

    grid_point_x_dimension, point_pdf = get_pdf(
        asarray((vector_0, vector_1)).T,
        plot=False,
        bandwidths=tuple(
            get_bandwidth(vector) * bandwidth_factor for vector in (vector_0, vector_1)
        ),
        grids=tuple(
            grid(vector.min(), vector.max(), 0.1, 24) for vector in (vector_0, vector_1)
        ),
    )

    grid_x, grid_y = get_grids(grid_point_x_dimension)

    pxy = reshape(point_pdf, (grid_x, grid_y))

    dx = grid_x[1] - grid_x[0]

    dy = grid_y[1] - grid_y[0]

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    pxpy = outer(px, py)

    mi = get_kld(pxy, pxpy).sum() * dx * dy

    return sqrt(1 - exp(-2 * mi)) * sign(r)
