from numpy import absolute, asarray, where, log, argmax, argmin

from .plot_plotly_figure import plot_plotly_figure


def compute_set_enrichment(
    element_score,
    set_elements,
    method="rank cdf ks",
    plot_data=False,
    plot_enrichment=True,
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
        "opacity": 0.8,
    }

    ########
    element_score = element_score.sort_values(ascending=False)

    ########
    if plot_data:

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
                        "marker": {"color": "#20d8ba"},
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

    ########
    r_h_i = where(r_h)[0]

    r_m_i = where(r_m)[0]

    ########
    r_h_v = r_h * absolute(element_score.values)

    r_h_p = r_h_v / r_h_v.sum()

    r_h_c = r_h_p.cumsum()

    ########
    r_m_v = r_m

    r_m_p = r_m_v / r_m_v.sum()

    r_m_c = r_m_p.cumsum()

    ########
    if plot_data:

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
            dimension_fraction_grid_extensions=(1 / 8,),
            dimension_n_grids=(64,),
            plot=False,
        )

    #######
    s_h_v = element_score.values[r_h_i]

    s_g, s_h_d = estimate_vector_density(s_h_v)

    s_h_p = s_h_d / s_h_d.sum()

    s_h_c = s_h_p.cumsum()

    #######
    s_m_v = element_score.values[r_m_i]

    s_g, s_m_d = estimate_vector_density(s_m_v)

    s_m_p = s_m_d / s_m_d.sum()

    s_m_c = s_m_p.cumsum()

    ########
    s_g = s_g.reshape(s_g.size)

    ########
    if plot_data:

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
                ],
            },
            None,
        )

    ########
    value, distribution, statistic = method.split()

    if value == "rank":

        if distribution == "pdf":

            x = r_h_p

            y = r_m_p

        elif distribution == "cdf":

            x = r_h_c

            y = r_m_c

    elif value == "score":

        if distribution == "pdf":

            x = s_h_p

            y = s_m_p

        elif distribution == "cdf":

            x = s_h_c

            y = s_m_c

    if statistic == "ks":

        signals = x - y

    else:

        x += 1e-6

        y += 1e-6

        if statistic == "kl":

            signals = x * log(x / y)

        elif statistic.startswith("js"):

            if statistic == "jsw":

                z = p_h * x + p_m * y

            else:

                z = (x + y) / 2

            z /= z.sum()

            z += 1e-6

            if statistic == "jsx":

                signals = x * log(x / z)

            elif statistic == "jsy":

                signals = y * log(y / z)

            elif statistic.endswith("0.5"):

                signals = x * log(x / z) * 0.5 + y * log(y / z) * 0.5

            elif statistic.endswith("p"):

                signals = x * log(x / z) * p_h + y * log(y / z) * p_m

            elif statistic == "jsw":

                signals = x * log(x / z) + (y - x) / 2

    if value == "score":

        signals = asarray(
            [signals[argmin(absolute(s_g - score))] for score in element_score.values]
        )

    enrichment = signals[argmax(absolute(signals))]

    ########
    if not plot_enrichment:

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
            "x": x,
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
        for i, (x, str_) in enumerate(zip(r_h_i, element_texts))
    ]

    ########
    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)

    return enrichment
