from numpy import apply_along_axis

from ..array import get_not_nan_unique


def _get_n_not_nan_unique(ve):

    return get_not_nan_unique(ve).size


def drop_constant(da):

    sh = da.shape

    print(sh)

    ax = int(sh[1] < sh[0])

    while True:

        ke_ = 1 < apply_along_axis(_get_n_not_nan_unique, ax, da.values)

        if ax == 0:

            da = da.iloc[:, ke_]

            ax = 1

        elif ax == 1:

            da = da.iloc[ke_, :]

            ax = 0

        if sh == da.shape:

            break

        else:

            sh = da.shape

            print(sh)

    return da
