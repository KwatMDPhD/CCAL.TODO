from numpy import (
    apply_along_axis,
    asarray,
    diff,
    full,
    integer,
    isnan,
    log as loge,
    log2,
    log10,
    logical_and,
    logical_not,
    logical_or,
    nan,
    quantile,
    sort,
    unique,
)
from numpy.random import seed, shuffle as shuffle_
from pandas import isna
from scipy.stats import rankdata

from .CONSTANT import RANDOM_SEED

# ==============================================================================
# 1D array
# ==============================================================================


def check_is_in(a, b):

    d = {v: None for v in b}

    return asarray([v in d for v in a])


def map_int(a):

    vti = {}

    itv = {}

    i = 0

    for v in a:

        if v not in vti:

            vti[v] = i

            itv[i] = v

            i += 1

    return vti, itv


# ==============================================================================
# v
# ==============================================================================


def check_is_all_sorted(v):

    d = diff(v)

    return (d <= 0).all() or (0 <= d).all()


def check_is_extreme(v, d, threshold_=None, n=None, std=None):

    v2 = v[~isnan(v)]

    if threshold_ is not None:

        l, h = threshold_

    elif n is not None:

        if n < 1:

            l = quantile(v2, n)

            h = quantile(v2, 1 - n)

        else:

            v2 = sort(v2)

            l = v2[n - 1]

            h = v2[-n]

    elif std is not None:

        m = v2.mean()

        e = v2.std() * std

        l = m - e

        h = m + e

    if d == "<>":

        return logical_or(v <= l, h <= v)

    elif d == "<":

        return v <= l

    elif d == ">":

        return h <= v


# ==============================================================================
# 2D array
# ==============================================================================


def shuffle(a, random_seed=RANDOM_SEED):

    b = a.copy()

    seed(random_seed)

    for i in range(b.shape[0]):

        shuffle_(b[i])

    return b


# ==============================================================================
# Matrix
# ==============================================================================


def function_on_2_2d_array(m0, m1, f):

    s0 = m0.shape[0]

    s1 = m1.shape[0]

    m = full((s0, s1), nan)

    for i0 in range(s0):

        v0 = m0[i0]

        for i1 in range(s1):

            m[i0, i1] = function(v0, m1[i1])

    return m


# ==============================================================================
# ND array
# ==============================================================================


def check_is_not_na(a):

    return logical_not(isna(a))


# ==============================================================================
# Number array
# ==============================================================================


def clip(a, s):

    m = a.mean()

    e = a.std() * s

    return a.clip(m - e, m + e)


def shift_min(a, m):

    if m == "0<":

        is_ = 0 < a

        if is_.any():

            m = a[is_].min()

        else:

            m = 1

    return m - a.min() + a


def log(a, base=2):

    return {2: log2, "e": loge, 10: log10}[base](a)


def normalize(a, m, rank_method="average"):

    if m == "-0-":

        return (a - a.mean()) / a.std()

    elif m == "0-1":

        n = a.min()

        return (a - n) / (a.max() - n)

    elif m == "sum":

        return a / a.sum()

    elif m == "rank":

        return rankdata(a, rank_method).reshape(a.shape)


def guess_type(a, max_n_category=16):

    if all(isinstance(n, integer) for n in a.ravel()):

        n = unique(a).size

        if n <= 2:

            return "binary"

        elif n <= max_n_category:

            return "categorical"

    return "continuous"


# ==============================================================================
# Number array with NaN
# ==============================================================================


def check_is_not_nan(a):

    return logical_not(isnan(a))


def get_not_nan_unique(a):

    return unique(a[check_is_not_nan(a)])


def function_on_1_a_not_nan(a, f, *a_, update=False, **k_):

    is_ = check_is_not_nan(a)

    r = f(a[is_], *a_, **k_)

    if update:

        b = full(a.shape, nan)

        b[is_] = r

        return b

    return r


def function_on_2_a_not_nan(a, b, f, *a_, **k_):

    is_ = logical_and(check_is_not_nan(a), check_is_not_nan(b))

    return f(a[is_], b[is_], *a_, **k_)
