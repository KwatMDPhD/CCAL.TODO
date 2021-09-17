from numpy import array, sum
from numpy.linalg import norm
from numpy.random import random_sample, seed

from ..constant import RANDOM_SEED


def _update_w(ma, maw, mah):

    return maw * (ma @ mah.T) / (maw @ mah @ mah.T)


def _update_h(ma, maw, mah):

    return mah * (maw.T @ ma) / (maw.T @ maw @ mah)


def _check_tolerable(er_it_ie, to):

    er2_, er1_ = array(er_it_ie)[-2:]

    return ((er2_ - er1_) / er2_ <= to).all()


def factorize(
    ma_,
    me,
    re,
    we_=None,
    to=1e-6,
    n_it=int(1e3),
    ra=RANDOM_SEED,
):

    n_ie = len(ma_)

    seed(seed=ra)

    no = norm(ma_[0])

    if we_ is None:

        we_ = [no / norm(ma) for ma in ma_]

    if me == "w":

        maw_ = [random_sample(size=[ma.shape[0], re]) for ma in ma_]

        mah = random_sample(size=[re, ma_[0].shape[1]])

        er_it_ie = [[norm(ma_[ie] - maw_[ie] @ mah) for ie in range(n_ie)]]

        for iei in range(n_it):

            nu = sum([we_[ie] * maw_[ie].T @ ma_[ie] for ie in range(n_ie)], axis=0)

            de = sum(
                [we_[ie] * maw_[ie].T @ maw_[ie] @ mah for ie in range(n_ie)], axis=0
            )

            mah *= nu / de

            maw_ = [_update_w(ma_[ie], maw_[ie], mah) for ie in range(n_ie)]

            er_it_ie.append([norm(ma_[ie] - maw_[ie] @ mah) for ie in range(n_ie)])

            if _check_tolerable(er_it_ie, to):

                break

        mah_ = [mah]

    elif me == "h":

        maw = random_sample(size=[ma_[0].shape[0], re])

        mah_ = [random_sample(size=[re, ma.shape[1]]) for ma in ma_]

        er_it_ie = [[norm(ma_[ie] - maw @ mah_[ie]) for ie in range(n_ie)]]

        for iei in range(n_it):

            nu = sum([we_[ie] * ma_[ie] @ mah_[ie].T for ie in range(n_ie)], axis=0)

            de = sum(
                [we_[ie] * maw @ mah_[ie] @ mah_[ie].T for ie in range(n_ie)], axis=0
            )

            maw *= nu / de

            mah_ = [_update_h(ma_[ie], maw, mah_[ie]) for ie in range(n_ie)]

            er_it_ie.append([norm(ma_[ie] - maw @ mah_[ie]) for ie in range(n_ie)])

            if _check_tolerable(er_it_ie, to):

                break

        maw_ = [maw]

    return maw_, mah_, array(er_it_ie).T
