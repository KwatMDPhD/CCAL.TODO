from numpy import absolute, array, sqrt, sum
from numpy.linalg import norm
from numpy.random import default_rng

from ..constant import FLOAT_RESOLUTION, RANDOM_SEED


def _initialize(sy, ma, re, rn):

    if sy == "w":

        si = [ma.shape[0], re]

    elif sy == "h":

        si = [re, ma.shape[1]]

    return absolute(rn.standard_normal(size=si)) * sqrt(ma.mean() / re)


def _clip(ma):

    ma[ma < FLOAT_RESOLUTION] = 0


def _update_w(ma, maw, mah):

    return maw * (ma @ mah.T) / (maw @ mah @ mah.T)


def _update_h(ma, maw, mah):

    return mah * (maw.T @ ma) / (maw.T @ maw @ mah)


def _check_tolerable(er_it_ie, to):

    er2_, er1_ = array(er_it_ie)[-2:]

    return ((er2_ - er1_) / er2_ <= to).all()


def factorize(ma_, me, re, we_=None, to=1e-6, n_it=int(1e3), ra=RANDOM_SEED):

    for ma in ma_:

        assert 0 <= ma.min()

    if we_ is None:

        si = ma_[0].size

        we_ = [si / ma.size for ma in ma_]

    n_ie = len(ma_)

    rn = default_rng(seed=ra)

    if me == "w":

        maw_ = [_initialize("w", ma, re, rn) for ma in ma_]

        mah = _initialize("h", ma_[0], re, rn)

        er_it_ie = [[norm(ma_[ie] - maw_[ie] @ mah) for ie in range(n_ie)]]

        for _ in range(n_it):

            nu = sum([we_[ie] * maw_[ie].T @ ma_[ie] for ie in range(n_ie)], axis=0)

            de = sum(
                [we_[ie] * maw_[ie].T @ maw_[ie] @ mah for ie in range(n_ie)], axis=0
            )

            mah *= nu / de

            _clip(mah)

            maw_ = [_update_w(ma_[ie], maw_[ie], mah) for ie in range(n_ie)]

            for ma in maw_:

                _clip(ma)

            er_it_ie.append([norm(ma_[ie] - maw_[ie] @ mah) for ie in range(n_ie)])

            if _check_tolerable(er_it_ie, to):

                break

        mah_ = [mah]

    elif me == "h":

        maw = _initialize("w", ma_[0], re, rn)

        mah_ = [_initialize("h", ma, re, rn) for ma in ma_]

        er_it_ie = [[norm(ma_[ie] - maw @ mah_[ie]) for ie in range(n_ie)]]

        for _ in range(n_it):

            nu = sum([we_[ie] * ma_[ie] @ mah_[ie].T for ie in range(n_ie)], axis=0)

            de = sum(
                [we_[ie] * maw @ mah_[ie] @ mah_[ie].T for ie in range(n_ie)], axis=0
            )

            maw *= nu / de

            _clip(maw)

            mah_ = [_update_h(ma_[ie], maw, mah_[ie]) for ie in range(n_ie)]

            for ma in mah_:

                _clip(ma)

            er_it_ie.append([norm(ma_[ie] - maw @ mah_[ie]) for ie in range(n_ie)])

            if _check_tolerable(er_it_ie, to):

                break

        maw_ = [maw]

    return maw_, mah_, array(er_it_ie).T
