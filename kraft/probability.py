from numpy import (
    absolute,
    apply_along_axis,
    log2,
    product,
    s_,
    unique,
)

from .grid import (
    get_1d_grid,
    get_1d_grid_resolution,
    plot as grid_plot,
)
from .kernel_density import (
    get_density,
)
from .plot import (
    plot_plotly,
)


def get_probability(
    nu_po_di,
    ba_=(),
    co__=(),
    pl=True,
    di_=(),
):

    (co_po_di, de_,) = get_density(
        nu_po_di,
        ba_=ba_,
        co__=co__,
        pl=pl,
        di_=di_,
    )

    pr_ = de_ / (
        de_.sum() * product([get_1d_grid_resolution(co_) for co_ in co_po_di.T])
    )

    if pl:

        grid_plot(
            co_po_di,
            pr_,
            di_=di_,
            nu="Probability",
        )

    return (
        co_po_di,
        pr_,
    )


def _get_probability(
    nu__,
):

    return nu__ / nu__.sum()


def get_posterior_probability(
    nu_po_di,
    ta=None,
    ba_=(),
    co__=(),
    pl=True,
    di_=(),
):

    (co_po_di, pr_,) = get_probability(
        nu_po_di,
        ba_=ba_,
        co__=co__,
        pl=pl,
        di_=di_,
    )

    ta_ = co_po_di[
        :,
        -1,
    ]

    pr__ = pr_.reshape([co_.size for co_ in get_1d_grid(co_po_di)])

    po__ = (
        apply_along_axis(
            _get_probability,
            -1,
            pr__,
        )
        * get_1d_grid_resolution(ta_)
    )

    po_ = po__.reshape(co_po_di.shape[0])

    if pl:

        grid_plot(
            co_po_di,
            po_,
            di_=di_,
            nu="Posterior Probability",
        )

    if ta is None:

        return (
            co_po_di,
            po_,
        )

    else:

        taco_ = unique(ta_)

        ie = absolute(taco_ - ta).argmin()

        ie_ = s_[ie :: taco_.size]

        co_po_dino = co_po_di[
            ie_,
            :-1,
        ]

        pono_ = po_[ie_]

        if pl:

            grid_plot(
                co_po_dino,
                pono_,
                di_=di_[:-1],
                nu="P({} = {:.2e} (~{}) | {})".format(
                    di_[-1],
                    taco_[ie],
                    ta,
                    *di_[:-1],
                ),
            )

        return (
            co_po_dino,
            pono_,
        )


def plot_nomogram(
    p_t0,
    p_t1,
    n_,
    p_t0__,
    p_t1__,
    pa="",
):

    l = {
        "title": {"text": "Nomogram"},
        "xaxis": {"title": {"text": "Log Odd Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "tickvals": tuple(range(1 + len(n_))),
            "ticktext": (
                "Prior",
                *n_,
            ),
        },
    }

    t = {"showlegend": False}

    d = [
        {
            "x": (
                0,
                log2(p_t1 / p_t0),
            ),
            "y": (0,) * 2,
            "marker": {"color": "#080808"},
            **t,
        },
    ]

    for (i, (n, p_t0_, p_t1_,),) in enumerate(
        zip(
            n_,
            p_t0__,
            p_t1__,
        )
    ):

        r_ = log2((p_t1_ / p_t0_) / (p_t1 / p_t0))

        plot_plotly(
            {
                "data": [
                    {
                        "name": "P(Target = 0)",
                        "y": p_t0_,
                    },
                    {
                        "name": "P(Target = 1)",
                        "y": p_t1_,
                    },
                    {
                        "name": "Log Odd Ratio",
                        "y": r_,
                    },
                ],
                "layout": {"title": {"text": n}},
            },
        )

        d.append(
            {
                "x": [
                    r_.min(),
                    r_.max(),
                ],
                "y": [1 + i] * 2,
                **t,
            },
        )

    plot_plotly(
        {
            "data": d,
            "layout": l,
        },
        pa=pa,
    )
