from numpy import asarray, exp, log, nan, outer, sign, sqrt, unique
from scipy.stats import pearsonr

from .array import normalize
from .grid import get_1d, make_1d, shape
from .kernel_density import get_bandwidth
from .probability import get_probability


def get_entropy(vector):

    probability_ = vector / vector.sum()

    return -(probability_ * log(probability_)).sum()


def get_kld(vector_0, vector_1):

    return vector_0 * log(vector_0 / vector_1)


def get_jsd(vector_0, vector_1, reference_vector=None):

    if reference_vector is None:

        reference_vector = (vector_0 + vector_1) / 2

    kld_0_ = get_kld(vector_0, reference_vector)

    kld_1_ = get_kld(vector_1, reference_vector)

    return kld_0_, kld_1_, kld_0_ - kld_1_


def get_zd(vector_0, vector_1):

    kld_0_ = get_kld(vector_0, vector_1)

    kld_1_ = get_kld(vector_1, vector_0)

    return kld_0_, kld_1_, kld_0_ - kld_1_


def get_ic(vector_0, vector_1):

    vector_0 = normalize(vector_0, "-0-")

    vector_1 = normalize(vector_1, "-0-")

    correlation = pearsonr(vector_0, vector_1)[0]

    bandwidth_factor = 1 - abs(correlation) * 2 / 3

    ########
    grid_nd, grid_nd_probabilities = get_probability(
        asarray((vector_0, vector_1)).T,
        plot=False,
        bandwidths=tuple(
            get_bandwidth(vector) * bandwidth_factor for vector in (vector_0, vector_1)
        ),
        grid_1ds=tuple(
            make_1d(vector.min(), vector.max(), 0.1, 24)
            for vector in (vector_0, vector_1)
        ),
    )

    grid_x, grid_y = get_1d(grid_nd)

    pxy = shape(grid_nd_probabilities, (grid_x, grid_y))

    dx = grid_x[1] - grid_x[0]

    dy = grid_y[1] - grid_y[0]

    px = pxy.sum(axis=1) * dy

    py = pxy.sum(axis=0) * dx

    pxpy = outer(px, py)

    mi = get_kld(pxy.ravel(), pxpy.ravel()).sum() * dx * dy

    return sqrt(1 - exp(-2 * mi)) * sign(correlation)
    ########


def get_icd(vector_0, vector_1):

    return (-get_ic(vector_0, vector_1) + 1) / 2
