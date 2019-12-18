from numpy import absolute, argmax, argmin, asarray, log, where

from .plot_plotly_figure import plot_plotly_figure


def compute_set_enrichment(
    element_score,
    set_elements,
    power=0,
    plot=True,
    title="Set Enrichment",
    element_score_name="Element Score",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    html_file_path=None,
):

    ########
    trace_tempalte = {
        "mode": "lines",
        "opacity": 0.64,
    }

    ########
    element_score = element_score.sort_values(ascending=False)

    ########
    if plot:

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": "Element Score"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Score"}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "y": element_score.values,
                        "marker": {"color": "#4e40d8"},
                        **trace_tempalte,
                    }
                ],
            },
            None,
        )

    ########
    set_element_ = {set_element: None for set_element in set_elements}

    ########
    r_h = asarray(
        [
            element_score_element in set_element_
            for element_score_element in element_score.index
        ],
        dtype=int,
    )

    r_m = 1 - r_h

    ########
    p_h = r_h.sum() / r_h.size

    p_m = r_m.sum() / r_m.size

    print(p_h, p_m)

    ########
    r_h_i = where(r_h)[0]

    r_m_i = where(r_m)[0]

    ########
    r_h_v = r_h * absolute(element_score.values) ** power

    r_h_p = r_h_v / r_h_v.sum()

    r_h_c = r_h_p.cumsum()

    ########
    r_m_v = r_m

    r_m_p = r_m_v / r_m_v.sum()

    r_m_c = r_m_p.cumsum()

    ########
    r_c_p = (r_h_p + r_m_p) / 2

    r_c_c = (r_h_c + r_m_c) / 2

    ########
    if plot:

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": "PDF(rank | hit-miss)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Hit", "y": r_h_p, **trace_tempalte},
                    {"type": "scatter", "name": "Miss", "y": r_m_p, **trace_tempalte},
                    {"type": "scatter", "name": "Center", "y": r_c_p, **trace_tempalte},
                ],
            },
            None,
        )

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": "CDF(rank | hit-miss)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Hit", "y": r_h_c, **trace_tempalte},
                    {"type": "scatter", "name": "Miss", "y": r_m_c, **trace_tempalte},
                    {"type": "scatter", "name": "Center", "y": r_c_c, **trace_tempalte},
                ],
            },
            None,
        )

    ########
    from .estimate_element_x_dimension_kernel_density import (
        estimate_element_x_dimension_kernel_density,
    )

    element_score_min = element_score.values.min()

    element_score_max = element_score.values.max()

    def estimate_vector_density(vector):

        return estimate_element_x_dimension_kernel_density(
            vector.reshape(vector.size, 1),
            dimension_grid_mins=(element_score_min,),
            dimension_grid_maxs=(element_score_max,),
            dimension_fraction_grid_extensions=(1e-16,),
            dimension_n_grids=(1024,),
            plot=False,
        )

    ########
    s_h_v = element_score.values[r_h_i]

    s_g, s_h_d = estimate_vector_density(s_h_v)

    s_h_p = s_h_d / s_h_d.sum()

    s_h_c = s_h_p.cumsum()

    ########
    s_m_v = element_score.values[r_m_i]

    s_g, s_m_d = estimate_vector_density(s_m_v)

    s_m_p = s_m_d / s_m_d.sum()

    s_m_c = s_m_p.cumsum()

    ########
    s_g = s_g.reshape(s_g.size)

    ########
    s_c_p = (s_h_p + s_m_p) / 2

    s_c_c = (s_h_c + s_m_c) / 2

    ########
    if plot:

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": "PDF(score | hit-miss)"},
                    "xaxis": {"title": {"text": "Score"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "name": "Hit",
                        "x": s_g,
                        "y": s_h_p,
                        **trace_tempalte,
                    },
                    {
                        "type": "scatter",
                        "name": "Miss",
                        "x": s_g,
                        "y": s_m_p,
                        **trace_tempalte,
                    },
                    {
                        "type": "scatter",
                        "name": "Center",
                        "x": s_g,
                        "y": s_c_p,
                        **trace_tempalte,
                    },
                ],
            },
            None,
        )

        plot_plotly_figure(
            {
                "layout": {
                    "title": {"text": "CDF(score | hit-miss)"},
                    "xaxis": {"title": {"text": "Score"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "name": "Hit",
                        "x": s_g,
                        "y": s_h_c,
                        **trace_tempalte,
                    },
                    {
                        "type": "scatter",
                        "name": "Miss",
                        "x": s_g,
                        "y": s_m_c,
                        **trace_tempalte,
                    },
                    {
                        "type": "scatter",
                        "name": "Center",
                        "x": s_g,
                        "y": s_c_c,
                        **trace_tempalte,
                    },
                ],
            },
            None,
        )

    ########
    str_signals = {}

    for (h, m, c, str_) in (
        # (r_h_p, r_m_p, r_c_p, "rank pdf"),
        # (r_h_c, r_m_c, r_c_c, "rank cdf"),
        (s_h_p, s_m_p, s_c_p, "score pdf"),
        (s_h_c, s_m_c, s_c_c, "score cdf"),
    ):

        ks = h - m

        str_signals["{} ks".format(str_)] = ks

        h += 1e-8

        m += 1e-8

        c += 1e-8

        str_signals["{} kl".format(str_)] = h * log(h / m)

        str_signals["{} jsh".format(str_)] = h * log(h / c)

        str_signals["{} jsm".format(str_)] = m * log(m / c)

        str_signals["{} js".format(str_)] = (
            str_signals["{} jsh".format(str_)] + str_signals["{} jsm".format(str_)]
        )

    ########
    for str_, signals in str_signals.items():

        if str_.startswith("score"):

            str_signals[str_] = asarray(
                [
                    signals[argmin(absolute(s_g - score))]
                    for score in element_score.values
                ]
            )

    ########
    plot_plotly_figure(
        {
            "layout": {
                "xaxis": {"title": {"text": "Rank"}},
                "yaxis": {"title": {"text": "Enrichment"}},
            },
            "data": [
                {"type": "scatter", "name": str_, "y": signals, **trace_tempalte}
                for str_, signals in str_signals.items()
            ],
        },
        None,
    )

    signals = str_signals["rank cdf ks"]

    enrichment = signals[argmax(absolute(signals))]

    ########
    if not plot:

        return enrichment

    ########
    y_fraction = 0.16

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {"anchor": "y", "title": "Rank"},
        "yaxis": {"domain": (0, y_fraction), "title": element_score_name},
        "yaxis2": {"domain": (y_fraction + 0.08, 1), "title": "Enrichment"},
    }

    ########
    data = []

    line_width = 2.4

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Cumulative Sum",
            "y": signals,
            "line": {"width": line_width, "color": "#20d9ba"},
            "fill": "tozeroy",
        }
    )

    ########
    r_h_i = where(r_h)[0]

    element_texts = element_score.index.values[r_h_i]

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Element",
            "x": r_h_i,
            "y": (0,) * r_h_i.size,
            "text": element_texts,
            "mode": "markers",
            "marker": {
                "symbol": "line-ns-open",
                "size": 8,
                "color": "#9017e6",
                "line": {"width": line_width / 2},
            },
            "hoverinfo": "x+text",
        }
    )

    ########
    data.append(
        {
            "type": "scatter",
            "name": "Element Score",
            "y": element_score.values,
            "text": element_score.index,
            "line": {"width": line_width, "color": "#4e40d8"},
            "fill": "tozeroy",
        }
    )

    ########
    layout["annotations"] = [
        {
            "x": h,
            "y": 0,
            "yref": "y2",
            "clicktoshow": "onoff",
            "text": "<b>{}</b>".format(str_),
            "showarrow": False,
            "font": {"size": annotation_text_font_size},
            "textangle": -90,
            "width": annotation_text_width,
            "borderpad": 0,
            "yshift": (-annotation_text_yshift, annotation_text_yshift)[i % 2],
        }
        for i, (h, str_) in enumerate(zip(r_h_i, element_texts))
    ]

    ########
    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)

    return enrichment
