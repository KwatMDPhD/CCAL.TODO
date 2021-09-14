from numpy import apply_along_axis, full, unique
from pandas import notna


def _check_has_enough_not_na(ve, n_no):

    return n_no <= notna(ve).sum()


def _check_has_enough_not_na_unique(ve, n_un):

    return n_un <= unique(ve[notna(ve)]).size


def drop(da, ax, n_no=None, n_un=None):

    sh = da.shape

    dr_ = full(sh[ax], True)

    if ax == 0:

        axa = 1

    elif ax == 1:

        axa = 0

    else:

        raise

    daa = da.values

    n_co = sh[1]

    if n_no is not None:

        if n_no == -1:

            n_no = n_co

        elif n_no < 1:

            n_no *= sh[axa]

        dr_ &= apply_along_axis(_check_has_enough_not_na, axa, daa, n_no)

    if n_un is not None:

        if n_un == -1:

            n_un = n_co

        elif n_un < 1:

            n_un *= sh[axa]

        dr_ &= apply_along_axis(_check_has_enough_not_na_unique, axa, daa, n_un)

    if ax == 0:

        da = da.loc[dr_, :]

    elif ax == 1:

        da = da.loc[:, dr_]

    print("{} => {}".format(sh, da.shape))

    return da
