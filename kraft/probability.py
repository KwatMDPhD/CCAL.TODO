from numpy import absolute, apply_along_axis, log2, product, s_, unique

from .grid import get_d, get_g1_, plot as grid_plot
from .kernel_density import get_density
from .plot import plot_plotly


def get_probability(pxd, plot=True, dimension_name_=None, **k_):

    gn, v = get_density(
        pxd,
        plot=plot,
        dimension_name_=dimension_name_,
        **k_,
    )

    v = v / (v.sum() * product([get_d(v) for v in gn.T]))

    if plot:

        grid_plot(
            gn,
            v,
            dimension_name_=dimension_name_,
            value_name="Probability",
        )

    return gn, v


def _get_probability(v):

    return v / v.sum()


def get_posterior_probability(pxd, target=None, plot=True, dimension_name_=None, **k_):

    gn, v = get_probability(
        pxd,
        plot=plot,
        dimension_name_=dimension_name_,
        **k_,
    )

    d = get_d(gn[:, -1])

    v = v.reshape([g1.size for g1 in get_g1_(gn)])

    v = apply_along_axis(_get_probability, -1, v) * d

    v = v.reshape(gn.shape[0])

    if plot:

        grid_plot(
            gn,
            v,
            dimension_name_=dimension_name_,
            value_name="Posterior Probability",
        )

    if target is not None:

        gt = unique(gn[:, -1])

        i = absolute(gt - target).argmin()

        i_ = s_[i :: gt.size]

        gn = gn[i_, :-1]

        v = v[i_]

        if plot:

            grid_plot(
                gn,
                v,
                dimension_name_=dimension_name_,
                value_name="P({} = {:.2e} (~{}) | {})".format(
                    dimension_name_[-1],
                    gt[i],
                    target,
                    *dimension_name_[:-1],
                ),
            )

    return gn, v


# TODO: rename
def plot_nomogram(p_t0, p_t1, n_, p_t0__, p_t1__, file_path=None):

    l = {
        "title": {"text": "Nomogram"},
        "xaxis": {"title": {"text": "Log Odd Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "tickvals": tuple(range(1 + len(n_))),
            "ticktext": ("Prior", *n_),
        },
    }

    t = {"showlegend": False}

    d = [
        {
            "x": (0, log2(p_t1 / p_t0)),
            "y": (0,) * 2,
            "marker": {"color": "#080808"},
            **t,
        }
    ]

    for i, (n, p_t0_, p_t1_) in enumerate(zip(n_, p_t0__, p_t1__)):

        r_ = log2((p_t1_ / p_t0_) / (p_t1 / p_t0))

        plot_plotly(
            {
                "data": [
                    {"name": "P(Target = 0)", "y": p_t0_},
                    {"name": "P(Target = 1)", "y": p_t1_},
                    {"name": "Log Odd Ratio", "y": r_},
                ],
                "layout": {"title": {"text": n}},
            },
        )

        d.append(
            {
                "x": [r_.min(), r_.max()],
                "y": [1 + i] * 2,
                **t,
            }
        )

    plot_plotly({"data": d, "layout": l}, file_path=file_path)
