from ..array import log
from ..plot import plot_plotly


def plot(pr1, pr2, na_, TODO1, TODO2, pa=""):

    n_da = len(na_)

    layout = {
        "title": {"text": "Nomogram"},
        "xaxis": {"title": {"text": "Log Odd Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "tickvals": list(range(1 + n_da)),
            "ticktext": ["Prior", *na_],
        },
    }

    trace = {"showlegend": False}

    data = [
        merge(
            trace,
            {"x": [0, log(pr2 / pr1)], "y": [0] * 2, "marker": {"color": "#080808"}},
        )
    ]

    for ie in range(n_da):

        po1_ = TODO1[ie]

        po2_ = TODO2[ie]

        ra_ = log((po2_ / po1_) / (pr2 / pr1))

        plot_plotly(
            {
                "data": [
                    {"name": "P(Target = 0)", "y": po1_},
                    {"name": "P(Target = 1)", "y": po2_},
                    {"name": "Log Odd Ratio", "y": ra_},
                ],
                "layout": {"title": {"text": na_[ie]}},
            }
        )

        data.append(merge(trace, {"x": [ra_.min(), ra_.max()], "y": [1 + ie] * 2}))

    plot_plotly({"data": data, "layout": layout}, pa=pa)
