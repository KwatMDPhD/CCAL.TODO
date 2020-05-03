from numpy import log2

from .plot_plotly import plot_plotly


def plot_nomogram(p_t1, p_t0, names, p_t10__, plot_=False):

    layout = {
        "title": {"text": "Nomogram"},
        "xaxis": {"title": {"text": "Log Odd Ratio"}},
        "yaxis": {
            "title": {"text": "Evidence"},
            "tickvals": tuple(range(1 + len(p_t10__))),
            "ticktext": ("Prior", *names),
        },
    }

    trace_ = {"mode": "lines"}

    monogram_trace_ = {"showlegend": False}

    data = [
        {
            "x": (0, log2(p_t1 / p_t0)),
            "y": (0,) * 2,
            "marker": {"color": "#080808"},
            **monogram_trace_,
        }
    ]

    for i, (name, (p_t1__, p_t0__)) in enumerate(zip(names, p_t10__)):

        log_odd_ratios = log2((p_t1__ / p_t0__) / (p_t1 / p_t0))

        if plot_:

            plot_plotly(
                {
                    "layout": {"title": {"text": name}},
                    "data": [
                        {"y": p_t1__, **trace_},
                        {"y": p_t0__, **trace_},
                        {"name": "Log Odd Ratio", "y": log_odd_ratios, **trace_},
                    ],
                },
            )

        data.append(
            {
                "x": (log_odd_ratios.min(), log_odd_ratios.max()),
                "y": (1 + i,) * 2,
                **monogram_trace_,
            }
        )

    plot_plotly({"layout": layout, "data": data})
