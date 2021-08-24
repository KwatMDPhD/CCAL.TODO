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

    nua_fe_sa = nu_fe_sa.values

    ro_ = nu_fe_sa.index.values

    co_ = nu_fe_sa.columns.values

    nar = nu_fe_sa.index.name

    nac = nu_fe_sa.columns.name

    si = nua_fe_sa.size

    if pl and si <= n_he:

        plot_heat_map(
            nu_fe_sa,
            layout={
                "title": title,
            },
        )

    bo_fe_sa = isnan(nua_fe_sa)

    n_na = bo_fe_sa.sum()

    if 0 < n_na:

        print("% NaN: {:.2%}".format(n_na / si))

        if pl:

            plot_histogram(
                [
                    Series(data=bo_fe_sa.sum(axis=1), index=ro_, name=nar),
                    Series(data=bo_fe_sa.sum(axis=0), index=co_, name=nac),
                ],
                layout={
                    "title": title,
                    "xaxis": {
                        "title": "N NaN",
                    },
                },
            )

    bo_fe_sa = logical_not(bo_fe_sa)

    nuan_fe_sa = nua_fe_sa[bo_fe_sa]

    print("(Not-NaN) min: {:.2e}".format(nuan_fe_sa.min()))

    print("(Not-NaN) median: {:.2e}".format(median(nuan_fe_sa)))

    print("(Not-NaN) mean: {:.2e}".format(nuan_fe_sa.mean()))

    print("(Not-NaN) max: {:.2e}".format(nuan_fe_sa.max()))

    if pl:

        plot_histogram(
            [
                Series(data=median(nuan_fe_sa, axis=1), index=ro_, name=nar),
                Series(data=median(nuan_fe_sa, axis=0), index=co_, name=nac),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": "(Not-NaN) Median",
                },
            },
        )

        la_ = array(
            ["{}_{}".format(*la_) for la_ in make_nd_grid([ro_, co_])[bo_fe_sa.ravel()]]
        )

        if n_hi < si:

            print("Choosing {} for histogram...".format(n_hi))

            ie_ = concatenate(
                [
                    choice(nuan_fe_sa.size, n_hi, False),
                    [nuan_fe_sa.argmin(), nuan_fe_sa.argmax()],
                ]
            )

            nuan_fe_sa = nuan_fe_sa[ie_]

            la_ = la_[ie_]

        plot_histogram(
            [
                Series(nuan_fe_sa, la_, name="All"),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": "(Not-NaN) Number",
                },
            },
        )
