from numpy import (
    full,
    integer,
    isnan,
    log as loge,
    log2,
    log10,
    logical_and,
    logical_not,
    logical_or,
    nan,
    quantile,
    sort,
    unique,
)
from scipy.stats import (
    rankdata,
)


def check_is_not_nan(
    nu__,
):

    return logical_not(isnan(nu__))


def get_not_nan_unique(
    nu__,
):

    return unique(nu__[check_is_not_nan(nu__)])


def clip(
    nu__,
    st,
):

    me = nu__.mean()

    st *= nu__.std()

    return nu__.clip(
        me - st,
        me + st,
    )


def normalize(
    nu__,
    me,
    ra="average",
):

    if me == "-0-":

        return (nu__ - nu__.mean()) / nu__.std()

    elif me == "0-1":

        mi = nu__.min()

        return (nu__ - mi) / (nu__.max() - mi)

    elif me == "sum":

        return nu__ / nu__.sum()

    elif me == "rank":

        return rankdata(
            nu__,
            ra,
        ).reshape(nu__.shape)


def shift_minimum(
    nu__,
    mi,
):

    if mi == "0<":

        bo__ = 0 < nu__

        if bo__.any():

            mi = nu__[bo__].min()

        else:

            mi = 1

        print("Shifting the minimum to {}...".format(mi))

    return nu__ + mi - nu__.min()


def log(
    nu__,
    ba=2,
):

    return {2: log2, "e": loge, 10: log10,}[
        ba
    ](nu__)


def guess_type(
    nu__,
    ma=16,
):

    if all(
        isinstance(
            nu,
            integer,
        )
        for nu in nu__.ravel()
    ):

        n_ca = unique(nu__).size

        if n_ca <= 2:

            return "binary"

        elif n_ca <= ma:

            return "categorical"

    return "continuous"


def check_is_extreme(
    nu__,
    di,
    th_=(),
    n_ex=0,
    st=0.0,
):

    nuno__ = nu__[check_is_not_nan(nu__)]

    if 0 < len(th_):

        (
            lo,
            hi,
        ) = th_

    elif 0 < n_ex:

        if n_ex < 1:

            lo = quantile(
                nuno__,
                n_ex,
            )

            hi = quantile(
                nuno__,
                1 - n_ex,
            )

        else:

            nuno__ = sort(
                nuno__,
                None,
            )

            lo = nuno__[n_ex - 1]

            hi = nuno__[-n_ex]

    elif 0 < st:

        me = nuno__.mean()

        st *= nuno__.std()

        lo = me - st

        hi = me + st

    if di == "<>":

        return logical_or(
            nu__ <= lo,
            hi <= nu__,
        )

    elif di == "<":

        return nu__ <= lo

    elif di == ">":

        return hi <= nu__


def apply_on_1(nu__, fu, *ar_, up=False, **ke_):

    bo__ = check_is_not_nan(nu__)

    re = fu(nu__[bo__], *ar_, **ke_)

    if up:

        nu2__ = full(
            nu__.shape,
            nan,
        )

        nu2__[bo__] = re

        return nu2__

    return re


def apply_on_2(nu1__, nu2__, fu, *ar_, **ke_):

    bo__ = logical_and(
        check_is_not_nan(nu1__),
        check_is_not_nan(nu2__),
    )

    return fu(nu1__[bo__], nu2__[bo__], *ar_, **ke_)


def apply_along_on_2(nu1_an_an, nu2_an_an, fu, *ar_, **ke_):

    n_ro1 = nu1_an_an.shape[0]

    n_ro2 = nu2_an_an.shape[0]

    nu3_an_an = full(
        [
            n_ro1,
            n_ro2,
        ],
        nan,
    )

    for ie1 in range(n_ro1):

        nu1_ = nu1_an_an[ie1]

        for ie2 in range(n_ro2):

            nu2_ = nu2_an_an[ie2]

            nu3_an_an[
                ie1,
                ie2,
            ] = fu(nu1_, nu2_, *ar_, **ke_)

    return nu3_an_an
