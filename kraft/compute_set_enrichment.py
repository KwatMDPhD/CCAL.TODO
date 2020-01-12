from numpy import absolute, asarray, isnan, log, nan, where

from .make_grid import make_grid
from .plot_plotly import plot_plotly


def compute_set_enrichment(
    element_score,
    set_elements,
    method,
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

    r_h = asarray(
        tuple(
            element_score_element in set_element_
            for element_score_element in element_score.index
        ),
        dtype=int,
    )

    if r_h.sum() == 0:

        return nan

    r_m = 1 - r_h

    #
    p_h = r_h.sum() / r_h.size

    p_m = r_m.sum() / r_m.size

    #
    r_h_i = where(r_h)[0]

    r_m_i = where(r_m)[0]

    #
    if method is None:

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
                        "mode": "markers",
                    },
                ],
            },
            None,
        )

    #
    if power != 0:

        r_h *= absolute(element_score.values) ** power

    #
    r_h_p = r_h / r_h.sum()

    r_m_p = r_m / r_m.sum()

    r_c_p = (r_h_p + r_m_p) / 2

    #
    def get_c(p):

        return p[::-1].cumsum()[::-1]

    r_h_c = get_c(r_h_p)

    r_m_c = get_c(r_m_p)

    r_c_c = (r_h_c + r_m_c) / 2

    #
    if method is None:

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
    if method is None:

        from .estimate_kernel_density import estimate_kernel_density

        s_g = make_grid(
            element_score.values.min(), element_score.values.max(), 1e-8, 64
        )

        element_score_g_index = asarray(
            tuple(absolute(s_g - score).argmin() for score in element_score.values)
        )

        def get_p_c(vector):

            point_x_dimension, kernel_densities = estimate_kernel_density(
                vector.reshape(vector.size, 1), grids=(s_g,), plot=False,
            )

            p = kernel_densities / (kernel_densities.sum() * (s_g[1] - s_g[0]))

            return p, get_c(p)

        #
        s_h_p, s_h_c = get_p_c(element_score.values[where(r_h)])

        s_m_p, s_m_c = get_p_c(element_score.values[where(r_m)])

        s_c_p = (s_h_p + s_m_p) / 2

        s_c_c = (s_h_c + s_m_c) / 2

        s_p, s_c = get_p_c(element_score.values)

        #
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
                    {"type": "scatter", "name": "All", "x": s_g, "y": s_p},
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
                    {"type": "scatter", "name": "All", "x": s_g, "y": s_c},
                ],
            },
            None,
        )

    #
    str_signals = {"rank cdf ks": r_h_c - r_m_c}

    #
    if method is None:

        #
        for (h, m, c, str_) in (
            # (r_h_c, r_m_c, r_c_c, "rank cdf"),
            (s_h_p, s_m_p, s_c_p, "score pdf"),
            # (s_h_c, s_m_c, s_c_c, "score cdf"),
        ):

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
            if str_.startswith("score "):

                #
                jsh_ = h * log(h / s_p)

                jsh_[isnan(jsh_)] = 0

                str_signals["{} jsh_".format(str_)] = jsh_

                #
                jsm_ = m * log(m / s_p)

                jsm_[isnan(jsm_)] = 0

                str_signals["{} jsm_".format(str_)] = jsm_

                #
                js_ = jsh_ * p_h + jsm_ * p_m

                str_signals["{} js_".format(str_)] = js_

        #
        for str_, signals in str_signals.items():

            if str_.startswith("score "):

                str_signals[str_] = signals[element_score_g_index]

        #
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

    if method is None:

        return

    #
    signals = str_signals[method]

    enrichment = signals[absolute(signals).argmax()]

    #
    if plot:

        #
        y_fraction = 0.16

        layout = {
            "title": {"text": title, "x": 0.5, "xanchor": "center"},
            "xaxis": {"anchor": "y", "title": "Rank"},
            "yaxis": {"domain": (0, y_fraction), "title": element_score_name},
            "yaxis2": {"domain": (y_fraction + 0.08, 1), "title": "Enrichment"},
        }

        #
        line_width = 2.4

        data = [
            {
                "type": "scatter",
                "name": "Element Score",
                "y": element_score.values,
                "text": element_score.index,
                "line": {"width": line_width, "color": "#4e40d8"},
                "fill": "tozeroy",
            },
            {
                "yaxis": "y2",
                "type": "scatter",
                "name": method,
                "y": signals,
                "line": {"width": line_width, "color": "#20d8ba"},
                "fill": "tozeroy",
            },
            {
                "yaxis": "y2",
                "type": "scatter",
                "name": "Element",
                "x": r_h_i,
                "y": (0,) * r_h_i.size,
                "text": element_score.index[r_h_i],
                "mode": "markers",
                "marker": {
                    "symbol": "line-ns-open",
                    "size": 8,
                    "color": "#9016e6",
                    "line": {"width": line_width / 2},
                },
                "hoverinfo": "x+text",
            },
        ]

        #
        layout["annotations"] = [
            {
                "x": r_h_i_,
                "y": 0,
                "yref": "y2",
                "clicktoshow": "onoff",
                "text": "<b>{}</b>".format(element_score.index[r_h_i_]),
                "showarrow": False,
                "font": {"size": annotation_text_font_size},
                "textangle": -90,
                "width": annotation_text_width,
                "borderpad": 0,
                "yshift": (-annotation_text_yshift, annotation_text_yshift)[i % 2],
            }
            for i, r_h_i_ in enumerate(r_h_i)
        ]

        #
        plot_plotly({"layout": layout, "data": data}, html_file_path)

    return enrichment
