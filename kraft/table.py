from numpy import apply_along_axis, full, nan, unique
from numpy.random import choice, seed
from pandas import DataFrame, Index, notna

from .any_ import map_integer
from .CONSTANT import RANDOM_SEED


def error(ta):

    for ie, la_ in enumerate(ta.axes):

        assert not la_.isna().any(), "Dimension {} ({}) has Na.".format(
            ie + 1, la_.name
        )

        assert not la_.has_duplicates, "Dimension {} ({}) is duplicated.".format(
            ie + 1, la_.name
        )


def count(ta):

    for la, se in ta.iteritems():

        print("-" * 80)

        print(la)

        an_co = se.value_counts()

        print(an_co)

        print("-" * 80)


def sync(ta_, ax):

    ta1 = ta_[0]

    la_ = ta1.axes[ax]

    for ta in ta_[1:]:

        la_ = la_.intersection(ta.axes[ax])

    la_ = sorted(la_)

    return [ta.reindex(la_, axis=ax) for ta in ta_]


def _check_has_enough_not_na(un_, n_no):

    return n_no <= notna(un_).sum()


def _check_has_enough_not_na_unique(nu_, n_un):

    return n_un <= unique(nu_[notna(nu_)]).size


def drop(ta, ax, n_no=None, n_un=None):

    assert not (n_no is None and n_un is None)

    sh = ta.shape

    bo_ = full(sh[ax], True)

    if ax == 0:

        axap = 1

    elif ax == 1:

        axap = 0

    taar = ta.to_numpy()

    if n_no is not None:

        if n_no < 1:

            n_no *= sh[axap]

        bo_ &= apply_along_axis(_check_has_enough_not_na, axap, taar, n_no)

    if n_un is not None:

        if n_un < 1:

            n_un *= sh[axap]

        bo_ &= apply_along_axis(_check_has_enough_not_na_unique, axap, taar, n_un)

    if ax == 0:

        ta = ta.loc[bo_, :]

    elif ax == 1:

        ta = ta.loc[:, bo_]

    print("{} => {}".format(sh, ta.shape))

    return ta


def drop2(ta, ax=None, **ke):

    sh = ta.shape

    if ax is None:

        ax = int(sh[0] < sh[1])

    re = False

    while True:

        ta = drop(ta, ax, **ke)

        sh2 = ta.shape

        if re and sh == sh2:

            return ta

        sh = sh2

        if ax == 0:

            ax = 1

        elif ax == 1:

            ax = 0

        re = True


def sample(ta, sh, ra=RANDOM_SEED, **ke):

    sh1, sh2 = ta.shape

    n_sa1, n_sa2 = sh

    seed(ra)

    if n_sa1 is not None:

        if n_sa1 < 1:

            n_sa1 = int(n_sa1 * sh1)

        ta = ta.iloc[choice(sh1, n_sa1, **ke), :]

    if n_sa2 is not None:

        if n_sa2 < 1:

            n_sa2 = int(n_sa2 * sh2)

        ta = ta.iloc[:, choice(sh2, n_sa2, **ke)]

    return ta


def map_to(ta, la, fu=None):

    ke_va = {}

    for ke_, va in zip(ta.to_numpy(), ta[la].to_numpy()):

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

    taar = full([len(la1_it), len(la2_it)], nan)

    for la1, la2, an in zip(la1_, la2_, an_):

        ie1 = la1_it[la1] - 1

        ie2 = la2_it[la2] - 1

        an0 = taar[ie1, ie2]

        if notna(an0) and callable(fu):

            an = fu(an0, an)

        taar[ie1, ie2] = an

    return DataFrame(
        taar,
        Index(la1_it, name=na1),
        Index(la2_it, name=na2),
    )
