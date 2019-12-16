from numpy import absolute, asarray, where, log

from .plot_plotly_figure import plot_plotly_figure


def compute_set_enrichment(
    element_score,
    set_elements,
    statistic="rcks",
    plot=True,
    title="Set Enrichment",
    element_score_name="Element Score",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    html_file_path=None,
):

    ########
    from .estimate_element_x_dimension_kernel_density import (
        estimate_element_x_dimension_kernel_density,
    )

    element_score_min = element_score.values.min()

    element_score_max = element_score.values.max()

    def estimate_vector_density(vector):

        return estimate_element_x_dimension_kernel_density(
            vector,
            dimension_grid_mins=(element_score_min,),
            dimension_grid_maxs=(element_score_max,),
            dimension_fraction_grid_extensions=(1 / 8,),
            dimension_n_grids=(2 ** 8,),
            plot=False,
        )

    trace_tempalte = {
        "mode": "lines",
        "opacity": 0.64,
    }

    ########
    set_element_ = {set_element: None for set_element in set_elements}

    ########
    r_h = asarray(
        [
            element_score_element in set_element_
            for element_score_element in element_score.index
        ],
        dtype=float,
    )

    r_m = 1 - r_h

    ########
    h_r_v = r_h * absolute(element_score.values)

    h_r_p = h_r_v / h_r_v.sum()

    h_r_c = h_r_p.cumsum()

    ########
    m_r_v = r_m

    m_r_p = m_r_v / m_r_v.sum()

    m_r_c = m_r_p.cumsum()

    ########
    plot_plotly_figure(
        {
            "layout": {"title": {"text": "PDF(rank | hit-miss)"}},
            "data": [
                {"type": "scatter", "y": h_r_p, **trace_tempalte},
                {"type": "scatter", "y": m_r_p, **trace_tempalte},
            ],
        },
        None,
    )

    plot_plotly_figure(
        {
            "layout": {"title": {"text": "CDF(rank | hit-miss)"}},
            "data": [
                {"type": "scatter", "y": h_r_c, **trace_tempalte},
                {"type": "scatter", "y": m_r_c, **trace_tempalte},
            ],
        },
        None,
    )

    #######
    h_s_v = element_score.values[where(r_h)]

    s_g, h_s_d = estimate_vector_density(h_s_v.reshape(h_s_v.size, 1))

    h_s_p = h_s_d / h_s_d.sum()

    h_s_c = h_s_p.cumsum()

    #######
    m_s_v = element_score.values[where(r_m)]

    s_g, m_s_d = estimate_vector_density(m_s_v.reshape(m_s_v.size, 1))

    m_s_p = m_s_d / m_s_d.sum()

    m_s_c = m_s_p.cumsum()

    ########
    s_g = s_g.reshape(s_g.size)

    ########
    plot_plotly_figure(
        {
            "layout": {"title": {"text": "PDF(score | hit-miss)"}},
            "data": [
                {"type": "scatter", "x": s_g, "y": h_s_p, **trace_tempalte},
                {"type": "scatter", "x": s_g, "y": m_s_p, **trace_tempalte},
            ],
        },
        None,
    )

    plot_plotly_figure(
        {
            "layout": {"title": {"text": "CDF(score | hit-miss)"}},
            "data": [
                {"type": "scatter", "x": s_g, "y": h_s_c, **trace_tempalte},
                {"type": "scatter", "x": s_g, "y": m_s_c, **trace_tempalte},
            ],
        },
        None,
    )

    ########
    if statistic == "rcks":  # GSEA, ssGSEA

        signals = h_r_c - m_r_c

    ########
    elif statistic == "spkl":  # KL_PDF

        s_g_signals = h_s_p * log(h_s_p / m_s_p)

        signals = asarray(
            [s_g_signals[absolute(s_g - score).argmin()] for score in element_score]
        )

    ########
    elif statistic == "scks":

        signals = h_s_c - m_s_c

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
