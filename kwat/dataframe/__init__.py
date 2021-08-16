from numpy import apply_along_axis, full, nan, unique
from numpy.random import choice, seed
from pandas import DataFrame, Index, notna

from .CONSTANT import RANDOM_SEED
from .iterable import map_integer


def error(da):

    for ie, la_ in enumerate(da.axes):

        ar_ = ie + 1, la_.name

        assert not la_.isna().any(), "Dimension {} ({}) has Na.".format(*ar_)

        assert not la_.has_duplicates, "Dimension {} ({}) is duplicated.".format(*ar_)


def count(da):

    for la, se in da.iteritems():

        print()

        print(la)

        print(se.value_counts())

        print()


def sync(da_, ax):

    la_ = da_[0].axes[ax]

    for da in da_[1:]:

        la_ = la_.intersection(da.axes[ax])

    la_ = sorted(la_)

    return [da.reindex(la_, axis=ax) for da in da_]


def _check_has_enough_not_na(ve, n_no):

    return n_no <= notna(ve).sum()


def _check_has_enough_not_na_unique(ve, n_un):

    return n_un <= unique(ve[notna(ve)]).size


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


def drop_until(da, ax=None, **ke):

    sh = da.shape

    if ax is None:

        ax = int(sh[0] < sh[1])

    re = False

    while True:

        da = drop(da, ax, **ke)

        sh2 = da.shape

        if re and sh == sh2:

            return da

        sh = sh2

        if ax == 0:

            ax = 1

        elif ax == 1:

            ax = 0

        re = True


def sample(da, sh, ra=RANDOM_SEED, **ke):

    si1, si2 = da.shape

    sa1, sa2 = sh

    seed(ra)

    if sa1 is not None:

        if sa1 < 1:

            sa1 = int(sa1 * si1)

        da = da.iloc[choice(si1, sa1, **ke), :]

    if sa2 is not None:

        if sa2 < 1:

            sa2 = int(sa2 * si2)

        da = da.iloc[:, choice(si2, sa2, **ke)]

    return da


def map_to(da, la, fu=None):

    ke_va = {}

    for ke_, va in zip(da.values, da[la].values):

        for ke in ke_:

            if fu is None:

                ke_va[ke] = va

            else:

                for ke in fu(ke):

                    ke_va[ke] = va

    return ke_va


def pivot(la1_, la2_, an_, na1="Dimension 1", na2="Dimension 2", fu=None):

    la1_it = map_integer(la1_)[0]

    la2_it = map_integer(la2_)[0]

    an_la1_la2 = full([len(la1_it), len(la2_it)], nan)

    for la1, la2, an in zip(la1_, la2_, an_):

        ie1 = la1_it[la1] - 1

        ie2 = la2_it[la2] - 1

        an0 = an_la1_la2[ie1, ie2]

        if notna(an0) and callable(fu):

            an = fu(an0, an)

        an_la1_la2[ie1, ie2] = an

    return DataFrame(
        an_la1_la2,
        Index(la1_it, name=na1),
        Index(la2_it, name=na2),
    )
