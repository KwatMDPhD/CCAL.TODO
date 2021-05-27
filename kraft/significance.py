from numpy import asarray, sqrt, where
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests


def get_moe(v, confidence=0.95):

    return norm.ppf(confidence) * v.std() / sqrt(v.size)


def get_p(n, r_, d):

    if d == "<":

        is_ = r_ <= n

    elif d == ">":

        is_ = n <= r_

    return max(1, is_.sum()) / r_.size


def get_p__q_(n_, r_, d, multipletests_method="fdr_bh"):

    if "<" in d:

        lp_ = asarray([get_p(n, r_, "<") for n in n_])

        lq_ = multipletests(lp_, method=multipletests_method)[1]

    if ">" in d:

        rp_ = asarray([get_p(n, r_, ">") for n in n_])

        rq_ = multipletests(rp_, method=multipletests_method)[1]

    if d == "<":

        return lp_, lq_

    elif d == ">":

        return rp_, rq_

    elif d == "<>":

        return where(lp_ < rp_, lp_, rp_), where(lq_ < rq_, lq_, rq_)
