from numpy import absolute, asarray, where, log

from .plot_plotly_figure import plot_plotly_figure


def compute_set_enrichment(
    element_score,
    set_elements,
    statistic,
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
        "opacity": 0.8,
    }

    ########
    element_score = element_score.sort_values(ascending=False)

    ########
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
                    "y": element_score,
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
    r_h_i = where(r_h)[0]

    r_m_i = where(r_m)[0]

    ########
    r_h_v = r_h

    r_h_p = r_h_v / r_h_v.sum()

    r_h_c = r_h_p.cumsum()

    ########
    r_m_v = r_m

    r_m_p = r_m_v / r_m_v.sum()

    r_m_c = r_m_p.cumsum()

    ########
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
            dimension_n_grids=(2 ** 8,),
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
    raise

    ########
    if statistic == "rcks":  # GSEA, ssGSEA

        signals = r_h_c - r_m_c

    ########
    elif statistic == "spkl":  # KL_PDF

        s_g_signals = s_h_p * log(s_h_p / s_m_p)

        signals = asarray(
            [s_g_signals[absolute(s_g - score).argmin()] for score in element_score]
        )

    ########
    elif statistic == "rckl":  # KL_CDF

        signals = r_h_c * log(r_h_c / r_m_c)

    ########
    elif statistic == "spjs0.5":  # JS_PDF

        s_p = (s_h_p + s_m_p) / 2

        s_p /= s_p.sum()

        s_g_signals = s_h_p * log(s_h_p / s_p) + s_m_p * log(s_m_p / s_p)

        signals = asarray(
            [s_g_signals[absolute(s_g - score).argmin()] for score in element_score]
        )

    ########
    elif statistic == "spjsp":  # JS_PDF_p

        s_p = (s_h_p + s_m_p) / 2

        s_p /= s_p.sum()

        s_g_signals = r_h.sum() / r_h.size * s_h_p * log(
            s_h_p / s_p
        ) + r_m.sum() / r_m.size * s_m_p * log(s_m_p / s_p)

        signals = asarray(
            [s_g_signals[absolute(s_g - score).argmin()] for score in element_score]
        )

    ########
    elif statistic == "spjsp":  # JS_PDF_p

        s_p = (s_h_p + s_m_p) / 2

        s_p /= s_p.sum()

        s_g_signals = r_h.sum() / r_h.size * s_h_p * log(
            s_h_p / s_p
        ) + r_m.sum() / r_m.size * s_m_p * log(s_m_p / s_p)

        signals = asarray(
            [s_g_signals[absolute(s_g - score).argmin()] for score in element_score]
        )

    ########
    elif statistic == "scks":

        signals = s_h_c - s_m_c

    ########
    if not plot:

        return

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

    element_texts = element_score[r_h_i].index

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
            "y": element_score,
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

    return
