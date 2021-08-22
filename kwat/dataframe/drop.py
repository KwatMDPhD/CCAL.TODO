from numpy import apply_along_axis, full


def drop(da, ax, n_no=None, n_un=None):

    assert not (n_no is None and n_un is None)

    sh = da.shape

    bo_ = full(sh[ax], True)

    if ax == 0:

        axap = 1

    elif ax == 1:

        axap = 0

    daar = da.values

    if n_no is not None:

        if n_no < 1:

            n_no *= sh[axap]

        bo_ &= apply_along_axis(_check_has_enough_not_na, axap, daar, n_no)

    if n_un is not None:

        if n_un < 1:

            n_un *= sh[axap]

        bo_ &= apply_along_axis(_check_has_enough_not_na_unique, axap, daar, n_un)

    if ax == 0:

        da = da.loc[bo_, :]

    elif ax == 1:

        da = da.loc[:, bo_]

    print("{} => {}".format(sh, da.shape))

    return da
