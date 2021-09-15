from numpy import apply_along_axis, full, unique
from pandas import notna


def _check_has_enough_not_na(nu_, n_no):

    return n_no <= notna(nu_).sum()


def _check_has_enough_not_na_unique(nu_, n_un):

    return n_un <= unique(nu_[notna(nu_)]).size


def drop(da, ax, n_no=None, n_un=None):

    sh = da.shape

    dr_ = full(sh[ax], True)

    if ax == 0:

        axa = 1

    elif ax == 1:

        axa = 0

    n_ma = sh[axa]

    dav = da.values

    if n_no is not None:

        if n_no == -1:

            n_no = n_ma

        elif n_no < 1:

            n_no *= n_ma

        dr_ &= apply_along_axis(_check_has_enough_not_na, axa, dav, n_no)

    if n_un is not None:

        if n_un == -1:

            n_un = n_ma

        elif n_un < 1:

            n_un *= n_ma

        dr_ &= apply_along_axis(_check_has_enough_not_na_unique, axa, dav, n_un)

    if ax == 0:

        dad = da.loc[dr_, :]

    elif ax == 1:

        dad = da.loc[:, dr_]

    print("{} => {}".format(sh, dad.shape))

    return dad
