from numpy import absolute, asarray, isnan, log, where

from .plot_plotly import plot_plotly


def compute_set_enrichment_2(
    element_score,
    set_elements,
    method="rank cdf ks",
    power=0,
    plot=True,
    title="Set Enrichment",
    element_score_name="Element Score",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    html_file_path=None,
):

    #
    element_score = element_score.sort_values()

    #
    set_element_ = {set_element: None for set_element in set_elements}

    #
    r_h = asarray(
        [
            element_score_element in set_element_
            for element_score_element in element_score.index
        ],
        dtype=int,
    )

    r_m = 1 - r_h

    #
    p_h = r_h.sum() / r_h.size

    p_m = r_m.sum() / r_m.size

    #
    r_h_i = where(r_h)[0]

    r_m_i = where(r_m)[0]

    #
    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Element Score"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Score"}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "name": "Hit ({:.3f})".format(p_h),
                        "x": r_h_i,
                        "y": element_score.values[r_h_i],
                        "text": element_score.index[r_h_i],
                        "mode": "markers",
                    },
                    {
                        "type": "scatter",
                        "name": "Miss ({:.3f})".format(p_m),
                        "x": r_m_i,
                        "y": element_score.values[r_m_i],
                        "text": element_score.index[r_m_i],
                    },
                ],
            },
            None,
        )

    #
    r_h_v = r_h * absolute(element_score.values) ** power

    r_h_p = r_h_v / r_h_v.sum()

    r_h_c = r_h_p[::-1].cumsum()[::-1]

    #
    r_m_v = r_m

    r_m_p = r_m_v / r_m_v.sum()

    r_m_c = r_m_p[::-1].cumsum()[::-1]

    #
    r_c_p = (r_h_p + r_m_p) / 2

    r_c_c = (r_h_c + r_m_c) / 2

    #
    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "PDF(rank | event)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Hit", "y": r_h_p},
                    {"type": "scatter", "name": "Miss", "y": r_m_p},
                    {"type": "scatter", "name": "Center", "y": r_c_p},
                ],
            },
            None,
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "CDF(rank | event)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Hit", "y": r_h_c},
                    {"type": "scatter", "name": "Miss", "y": r_m_c},
                    {"type": "scatter", "name": "Center", "y": r_c_c},
                ],
            },
            None,
        )

    #
    from .estimate_kernel_density import (
        estimate_kernel_density,
    )

    element_score_min = element_score.values.min()

    element_score_max = element_score.values.max()

    def estimate_vector_density(vector):

        return estimate_kernel_density(
            vector.reshape(vector.size, 1),
            dimension_grid_mins=(element_score_min,),
            dimension_grid_maxs=(element_score_max,),
            dimension_fraction_grid_extensions=(1e-8,),
            dimension_n_grids=(1e3,),
            plot=False,
        )

    #
    s_g, s_d = estimate_vector_density(element_score.values)

    s_g = s_g.reshape(s_g.size)

    s_p = s_d / s_d.sum()

    s_c = s_p[::-1].cumsum()[::-1]

    #
    s_h_v = element_score.values[where(r_h)]

    s_h_d = estimate_vector_density(s_h_v)[1]

    s_h_p = s_h_d / s_h_d.sum()

    s_h_c = s_h_p[::-1].cumsum()[::-1]

    #
    s_m_v = element_score.values[where(r_m)]

    s_m_d = estimate_vector_density(s_m_v)[1]

    s_m_p = s_m_d / s_m_d.sum()

    s_m_c = s_m_p[::-1].cumsum()[::-1]

    #
    s_c_p = (s_h_p + s_m_p) / 2

    s_c_c = (s_h_c + s_m_c) / 2

    #
    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "PDF(score | event)"},
                    "xaxis": {"title": {"text": "Score"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Hit", "x": s_g, "y": s_h_p},
                    {"type": "scatter", "name": "Miss", "x": s_g, "y": s_m_p},
                    {"type": "scatter", "name": "Center", "x": s_g, "y": s_c_p},
                    {"type": "scatter", "name": "Hit | Miss (fit)", "x": s_g, "y": s_p},
                    {
                        "type": "scatter",
                        "name": "Hit | Miss (weight)",
                        "x": s_g,
                        "y": s_h_p * p_h + s_m_p * p_m,
                    },
                ],
            },
            None,
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "CDF(score | event)"},
                    "xaxis": {"title": {"text": "Score"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Hit", "x": s_g, "y": s_h_c},
                    {"type": "scatter", "name": "Miss", "x": s_g, "y": s_m_c},
                    {"type": "scatter", "name": "Center", "x": s_g, "y": s_c_c},
                    {"type": "scatter", "name": "Hit | Miss", "x": s_g, "y": s_c},
                ],
            },
            None,
        )

    #
    str_signals = {}

    for (h, m, c, str_) in (
        # (r_h_c, r_m_c, r_c_c, "rank cdf"),
        (s_h_p, s_m_p, s_c_p, "score pdf"),
        # (s_h_c, s_m_c, s_c_c, "score cdf"),
    ):

        # #
        # ks = h - m

        # str_signals["{} ks".format(str_)] = ks

        #
        jsh = h * log(h / c)

        jsh[isnan(jsh)] = 0

        str_signals["{} jsh".format(str_)] = jsh

        #
        jsm = m * log(m / c)

        jsm[isnan(jsm)] = 0

        str_signals["{} jsm".format(str_)] = jsm

        #
        js = (jsh + jsm) / 2

        str_signals["{} js".format(str_)] = js

        #
        r = s_p

        #
        jsh_ = h * log(h / r)

        jsh_[isnan(jsh_)] = 0

        str_signals["{} jsh_".format(str_)] = jsh_

        #
        jsm_ = m * log(m / r)

        jsm_[isnan(jsm_)] = 0

        str_signals["{} jsm_".format(str_)] = jsm_

        #
        js_ = jsh_ * p_h + jsm_ * p_m

        str_signals["{} js_".format(str_)] = js_

    #
    for str_, signals in str_signals.items():

        if str_.startswith("score "):

            str_signals[str_] = asarray(
                [
                    signals[absolute(s_g - score).argmin()]
                    for score in element_score.values
                ]
            )

    #
    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Statistics"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Enrichment"}},
                },
                "data": [
                    {"type": "scatter", "name": str_, "y": signals}
                    for str_, signals in str_signals.items()
                ],
            },
            None,
        )

    signals = str_signals[method]

    enrichment = signals[absolute(signals).argmax()]

    #
    if not plot:

        return enrichment

    #
    y_fraction = 0.16

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {"anchor": "y", "title": "Rank"},
        "yaxis": {"domain": (0, y_fraction), "title": element_score_name},
        "yaxis2": {"domain": (y_fraction + 0.08, 1), "title": "Enrichment"},
    }

    #
    data = []

    line_width = 2.4

    #
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

    #
    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": method,
            "y": signals,
            "line": {"width": line_width, "color": "#20d8ba"},
            "fill": "tozeroy",
        }
    )

    #
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
                "color": "#9016e6",
                "line": {"width": line_width / 2},
            },
            "hoverinfo": "x+text",
        }
    )

    layout["annotations"] = [
        {
            "x": r_h_i_,
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
        for i, (r_h_i_, str_) in enumerate(zip(r_h_i, element_texts))
    ]

    #
    plot_plotly({"layout": layout, "data": data}, html_file_path)

    return enrichment
