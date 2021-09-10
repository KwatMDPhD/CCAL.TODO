from numpy import array, sum
from numpy.linalg import norm
from numpy.random import random_sample, seed

from ..constant import random_seed
from ._is_tolerable import _is_tolerable
from ._update_h import _update_h
from ._update_w import _update_w


def factorize(
    ma_,
    mo,
    re,
    we_=None,
    to=1e-6,
    n_it=int(1e3),
    ra=random_seed,
):

    n_ma = len(ma_)

    seed(seed=ra)

    no = norm(ma_[0])

    if we_ is None:

        we_ = [no / norm(ma) for ma in ma_]

    if mo == "wm":

        wm_ = [random_sample(size=[ma.shape[0], re]) for ma in ma_]

        hm = random_sample(size=[re, ma_[0].shape[1]])

        er_ = [[norm(ma_[ie] - wm_[ie] @ hm) for ie in range(n_ma)]]

        for iei in range(n_it):

            nu = sum([we_[ie] * wm_[ie].T @ ma_[ie] for ie in range(n_ma)], axis=0)

            de = sum([we_[ie] * wm_[ie].T @ wm_[ie] @ hm for ie in range(n_ma)], axis=0)

            hm *= nu / de

            wm_ = [_update_w(ma_[ie], wm_[ie], hm) for ie in range(n_ma)]

            er_.append([norm(ma_[ie] - wm_[ie] @ hm) for ie in range(n_ma)])

            if _is_tolerable(er_, to):

                break

        hm_ = [hm]

    elif mo == "hm":

        wm = random_sample(size=[ma_[0].shape[0], re])

        hm_ = [random_sample(size=[re, ma.shape[1]]) for ma in ma_]

        er_ = [[norm(ma_[ie] - wm @ hm_[ie]) for ie in range(n_ma)]]

        for iei in range(n_it):

            nu = sum([we_[ie] * ma_[ie] @ hm_[ie].T for ie in range(n_ma)], axis=0)

            de = sum([we_[ie] * wm @ hm_[ie] @ hm_[ie].T for ie in range(n_ma)], axis=0)

            wm *= nu / de

            hm_ = [_update_h(ma_[ie], wm, hm_[ie]) for ie in range(n_ma)]

            er_.append([norm(ma_[ie] - wm @ hm_[ie]) for ie in range(n_ma)])

            if _is_tolerable(er_, to):

                break

        wm_ = [wm]

    return wm_, hm_, array(er_).T
