from numpy import array, concatenate, isnan, logical_not, median, nanmedian
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

    nua_fe_sa = nu_fe_sa.values

    ro_ = nu_fe_sa.index.values

    ron = nu_fe_sa.index.name

    co_ = nu_fe_sa.columns.values

    con = nu_fe_sa.columns.name

    si = nua_fe_sa.size

    if pl and si <= n_he:

        plot_heat_map(
            nu_fe_sa,
            layout={
                "title": title,
            },
        )

    na_fe_sa = isnan(nua_fe_sa)

    n_na = na_fe_sa.sum()

    if 0 < n_na:

        print("% NaN: {:.2%}".format(n_na / si))

        if pl:

            plot_histogram(
                [
                    Series(data=na_fe_sa.sum(axis=1), index=ro_, name=ron),
                    Series(data=na_fe_sa.sum(axis=0), index=co_, name=con),
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
                Series(data=nanmedian(nua_fe_sa, axis=1), index=ro_, name=ron),
                Series(data=nanmedian(nua_fe_sa, axis=0), index=co_, name=con),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": "(Not-NaN) Median",
                },
            },
        )

    go_fe_sa = logical_not(na_fe_sa)

    go_ = nua_fe_sa[go_fe_sa]

    print("(Not-NaN) min: {:.2e}".format(go_.min()))

    print("(Not-NaN) median: {:.2e}".format(median(go_)))

    print("(Not-NaN) mean: {:.2e}".format(go_.mean()))

    print("(Not-NaN) max: {:.2e}".format(go_.max()))

    if pl:

        la_ = array(
            ["{}_{}".format(*la_) for la_ in make_nd_grid([ro_, co_])[go_fe_sa.ravel()]]
        )

        if n_hi < si:

            print("Choosing {} for histogram...".format(n_hi))

            ie_ = concatenate(
                [
                    choice(go_.size, n_hi, False),
                    [go_.argmin(), go_.argmax()],
                ]
            )

            go_ = go_[ie_]

            la_ = la_[ie_]

        plot_histogram(
            [
                Series(data=go_, index=la_, name="All"),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": "(Not-NaN) Number",
                },
            },
        )
