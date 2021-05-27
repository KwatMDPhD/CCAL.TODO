from numpy import asarray, exp, log, nan, outer, sign, sqrt, unique
from scipy.stats import pearsonr

from .array import normalize
from .grid import get_g1_, make_g1
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

    if unique(vector_0).size == 1 or unique(vector_1).size == 1:

        return nan

    vector_0 = normalize(vector_0, "-0-")

    vector_1 = normalize(vector_1, "-0-")

    correlation = pearsonr(vector_0, vector_1)[0]

    bandwidth_factor = 1 - abs(correlation) * 2 / 3

    nd_grid, nd_probability_vector = get_probability(
        asarray((vector_0, vector_1)).T,
        plot=False,
        bandwidth_=tuple(
            get_bandwidth(vector) * bandwidth_factor for vector in (vector_0, vector_1)
        ),
        _1d_grid_=tuple(
            make_g1(vector.min(), vector.max(), 0.1, 24)
            for vector in (vector_0, vector_1)
        ),
    )

    axis_0_grid, axis_1_grid = get_g1_(nd_grid)

    p01 = nd_probability_vector.reshape((axis_0_grid.size, axis_1_grid.size))

    d0 = axis_0_grid[1] - axis_0_grid[0]

    d1 = axis_1_grid[1] - axis_1_grid[0]

    p0 = p01.sum(axis=1) * d1

    p1 = p01.sum(axis=0) * d0

    p0p1 = outer(p0, p1)

    mi = get_kld(p01.ravel(), p0p1.ravel()).sum() * d0 * d1

    return sqrt(1 - exp(-2 * mi)) * sign(correlation)


def get_icd(vector_0, vector_1):

    return (-get_ic(vector_0, vector_1) + 1) / 2
