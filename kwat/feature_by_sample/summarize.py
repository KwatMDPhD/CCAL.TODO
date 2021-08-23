from numpy import array, concatenate, isnan, logical_not, median
from numpy.random import choice
from pandas import Series

from ..grid import make_nd_grid
from ..plot import plot_heat_map, plot_histogram


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

        la_ = array(
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
