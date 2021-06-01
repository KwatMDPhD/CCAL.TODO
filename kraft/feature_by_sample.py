from numpy import apply_along_axis, asarray, concatenate, isnan, logical_not, median
from numpy.random import choice
from pandas import DataFrame, Series, concat

from .grid import make_nd_grid
from .object.array import guess_type, log, normalize, shift
from .object.dataframe import drop, drop_until
from .object.series import binarize
from .plot import plot_heat_map, plot_histogram
from .python import cast_builtin


def collapse(nu_fe_sa):

    print(nu_fe_sa.shape)

    print("Collapsing...")

    nu_fe_sa = nu_fe_sa.groupby(level=0).median()

    print(nu_fe_sa.shape)

    return nu_fe_sa


def summarize(
    nu_fe_sa,
    pl=True,
    title="Name",
    n_he=int(1e6),
    n_hi=int(1e3),
):

    print(nu_fe_sa.shape)

    nuar_fe_sa = nu_fe_sa.values

    la1_ = nu_fe_sa.index.values

    la2_ = nu_fe_sa.columns.values

    na1 = nu_fe_sa.index.name

    na2 = nu_fe_sa.columns.name

    si = nuar_fe_sa.size

    if pl and si <= n_he:

        plot_heat_map(
            nu_fe_sa,
            layout={
                "title": title,
            },
        )

    bo_fe_sa = isnan(nuar_fe_sa)

    n_na = bo_fe_sa.sum()

    if 0 < n_na:

        print("% NaN: {:.2%}".format(n_na / si))

        if pl:

            plot_histogram(
                [
                    Series(bo_fe_sa.sum(axis=1), la1_, name=na1),
                    Series(bo_fe_sa.sum(axis=0), la2_, name=na2),
                ],
                layout={
                    "title": title,
                    "xaxis": {
                        "title": "N NaN",
                    },
                },
            )

    if pl:

        plot_histogram(
            [
                Series(median(nuar_fe_sa, axis=1), la1_, name=na1),
                Series(median(nuar_fe_sa, axis=0), la2_, name=na2),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": "(Not-NaN) Median",
                },
            },
        )

    bo_fe_sa = logical_not(bo_fe_sa)

    nuarbo_fe_sa = nuar_fe_sa[bo_fe_sa]

    print("(Not-NaN) min: {:.2e}".format(nuarbo_fe_sa.min()))

    print("(Not-NaN) median: {:.2e}".format(median(nuarbo_fe_sa)))

    print("(Not-NaN) mean: {:.2e}".format(nuarbo_fe_sa.mean()))

    print("(Not-NaN) max: {:.2e}".format(nuarbo_fe_sa.max()))

    if pl:

        la_ = asarray(
            [
                "{}_{}".format(*la_)
                for la_ in make_nd_grid([la1_, la2_])[bo_fe_sa.ravel()]
            ]
        )

        if n_hi < nuarbo_fe_sa.size:

            print("Choosing {} for histogram...".format(n_hi))

            ie_ = concatenate(
                [
                    choice(nuarbo_fe_sa.size, n_hi, False),
                    [nuarbo_fe_sa.argmin(), nuarbo_fe_sa.argmax()],
                ]
            )

            nuarbo_fe_sa = nuarbo_fe_sa[ie_]

            la_ = la_[ie_]

        plot_histogram(
            [
                Series(nuarbo_fe_sa, la_, name="All"),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": "(Not-NaN) Number",
                },
            },
        )


def process(
    nu_fe_sa,
    fe_=(),
    sa_=(),
    na=None,
    axdr=None,
    n_no=None,
    n_un=None,
    lo=None,
    mish=None,
    me=None,
    axno=None,
    mi=None,
    ma=None,
    **ke,
):

    if 0 < len(fe_):

        print("Dropping {}: {}...".format(nu_fe_sa.index.name, fe_))

        nu_fe_sa = nu_fe_sa.drop(fe_, errors="ignore")

        summarize(nu_fe_sa, **ke)

    if 0 < len(sa_):

        print("Dropping {}: {}...".format(nu_fe_sa.columns.name, sa_))

        nu_fe_sa = nu_fe_sa.drop(sa_, 1, errors="ignore")

        summarize(nu_fe_sa, **ke)

    if na is not None:

        print("NaNizing <= {}...".format(na))

        nu_fe_sa[nu_fe_sa <= na] = nan

        summarize(nu_fe_sa, **ke)

    if n_no is not None or n_un is not None:

        print("Dropping (axdr={}, n_no={}, n_un={})...".format(axdr, n_no, n_un))

        if axdr is None:

            dr = drop_until

        else:

            dr = drop

        sh = nu_fe_sa.shape

        nu_fe_sa = dr(
            nu_fe_sa,
            axdr,
            n_no=n_no,
            n_un=n_un,
        )

        if sh != nu_fe_sa.shape:

            summarize(nu_fe_sa, **ke)

    if lo is not None:

        print("Logging (mish={}, lo={})...".format(mish, lo))

        nuar_fe_sa = nu_fe_sa.values

        if mish is not None:

            nuar_fe_sa = shift(nuar_fe_sa, mish)

        nu_fe_sa = DataFrame(
            log(nuar_fe_sa, ba=lo),
            nu_fe_sa.index,
            nu_fe_sa.columns,
        )

        summarize(nu_fe_sa, **ke)

    if me is not None:

        print("Axis-{} {} normalizing...".format(axno, me))

        nu_fe_sa = DataFrame(
            apply_along_axis(normalize, axno, nu_fe_sa.values, me),
            nu_fe_sa.index,
            nu_fe_sa.columns,
        )

        summarize(nu_fe_sa, **ke)

    if mi is not None or ma is not None:

        print("Clipping |{} - {}|...".format(mi, ma))

        nu_fe_sa = nu_fe_sa.clip(mi, ma)

        summarize(nu_fe_sa, **ke)

    return nu_fe_sa


def separate_type(nu_fe_sa, dr=True, pr=True):

    co__ = []

    bi_in_sa_ = []

    for _, nu_ in nu_fe_sa.iterrows():

        if dr and nu_.unique().size == 1:

            continue

        try:

            bo = guess_type(nu_.dropna().astype(float).values) == "continuous"

        except ValueError:

            bo = False

        if bo:

            co__.append(nu_.apply(cast_builtin))

        else:

            bi_in_sa = binarize(nu_)

            if pr:

                te = "{}.{{}}".format(bi_in_sa.index.name)

            else:

                te = "{}"

            bi_in_sa.index = [te.format(la) for la in bi_in_sa.index]

            bi_in_sa_.append(bi_in_sa)

    te = "{} ({{}})".format(nu_fe_sa.index.name)

    if 0 < len(co__):

        co_in_sa = DataFrame(co__)

        co_in_sa.index.name = te.format("continuous")

    else:

        co_in_sa = None

    if 0 < len(bi_in_sa_):

        bi_in_sa = concat(bi_in_sa_)

        bi_in_sa.index.name = te.format("binary")

    else:

        bi_in_sa = None

    return co_in_sa, bi_in_sa
