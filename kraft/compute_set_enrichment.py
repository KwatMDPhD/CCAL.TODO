from numpy import absolute, asarray, log, nan, where

from .make_grid import make_grid
from .plot_plotly import plot_plotly


def compute_set_enrichment(
    element_score,
    set_elements,
    method="rank cdf ks",
    power=0,
    n_grid=64,
    plot=True,
    title="Set Enrichment",
    element_score_name="Element Score",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    html_file_path=None,
):

    element_score = element_score.sort_values()

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

    p_h = r_h.sum() / r_h.size

    p_m = r_m.sum() / r_m.size

    r_h_i = where(r_h)[0]

    r_m_i = where(r_m)[0]

    if method != "rank cdf ks":

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
                        "name": "Miss ({:.3f})".format(p_m),
                        "x": r_m_i,
                        "y": element_score.values[r_m_i],
                        "text": element_score.index[r_m_i],
                        "mode": "markers",
                    },
                    {
                        "type": "scatter",
                        "name": "Hit ({:.3f})".format(p_h),
                        "x": r_h_i,
                        "y": element_score.values[r_h_i],
                        "text": element_score.index[r_h_i],
                        "mode": "markers",
                    },
                ],
            },
            None,
        )

    if power != 0:

        r_h *= absolute(element_score.values) ** power

    r_h_p = r_h / r_h.sum()

    r_m_p = r_m / r_m.sum()

    def get_c(p):

        return p[::-1].cumsum()[::-1]

    r_h_c = get_c(r_h_p)

    r_m_c = get_c(r_m_p)

    if method != "rank cdf ks":

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "PDF(rank | event)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Miss", "y": r_m_p},
                    {"type": "scatter", "name": "Hit", "y": r_h_p},
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
                    {"type": "scatter", "name": "Miss", "y": r_m_c},
                    {"type": "scatter", "name": "Hit", "y": r_h_c},
                ],
            },
            None,
        )

    if method != "rank cdf ks":

        from .compute_bandwidth import compute_bandwidth
        from .estimate_kernel_density import estimate_kernel_density

        s_b = compute_bandwidth(element_score.values)

        s_g = make_grid(
            element_score.values.min(), element_score.values.max(), 1 / 10, n_grid
        )

        def get_p_c(vector):

            point_x_dimension, kernel_densities = estimate_kernel_density(
                vector.reshape(vector.size, 1),
                bandwidths=(s_b,),
                grids=(s_g,),
                plot=False,
            )

            p = kernel_densities / (kernel_densities.sum() * (s_g[1] - s_g[0]))

            return p, get_c(p)

        s_h_p, s_h_c = get_p_c(element_score.values[where(r_h)])

        s_m_p, s_m_c = get_p_c(element_score.values[where(r_m)])

        s_p, s_c = get_p_c(element_score.values)

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "PDF(score | event)"},
                    "xaxis": {"title": {"text": "Score"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {"type": "scatter", "name": "Miss", "x": s_g, "y": s_m_p},
                    {"type": "scatter", "name": "Hit", "x": s_g, "y": s_h_p},
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
                    {"type": "scatter", "name": "Miss", "x": s_g, "y": s_m_c},
                    {"type": "scatter", "name": "Hit", "x": s_g, "y": s_h_c},
                    {"type": "scatter", "name": "All", "x": s_g, "y": s_c},
                ],
            },
            None,
        )

    str_signals = {"rank cdf ks": r_h_c - r_m_c}

    if method != "rank cdf ks":

        jsh = s_h_p * log(s_h_p / s_p)

        jsh[jsh == nan] = 0

        str_signals["score pdf h"] = jsh

        jsm = s_m_p * log(s_m_p / s_p)

        jsm[jsm == nan] = 0

        str_signals["score pdf m"] = jsm

        str_signals["score pdf k"] = p_h * jsh - p_m * jsm

        element_score_g_index = asarray(
            tuple(absolute(s_g - score).argmin() for score in element_score.values)
        )

        str_signals["score cdf m"] = None

        str_signals["score cdf k"] = None

        for str_, signals in str_signals.items():

            if str_.startswith("score "):

                str_signals[str_] = signals[element_score_g_index]

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

    if plot:

        y_fraction = 0.16

        layout = {
            "title": {"text": title, "x": 0.5, "xanchor": "center"},
            "xaxis": {"anchor": "y", "title": "Rank"},
            "yaxis": {"domain": (0, y_fraction), "title": element_score_name},
            "yaxis2": {"domain": (y_fraction + 0.08, 1), "title": "Enrichment"},
        }

        line_width = 2

        data = [
            {
                "type": "scatter",
                "name": "Element Score",
                "y": element_score.values,
                "text": element_score.index,
                "line": {"width": line_width, "color": "#9016e6"},
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
                    "color": "#4e40d8",  # "#2e211b",
                    "line": {"width": line_width * 0.64},
                },
                "hoverinfo": "x+text",
            },
        ]

        for is_, color in (
            (signals < 0, "#0088ff"),
            (0 < signals, "#ff1968"),
        ):

            data.append(
                {
                    "yaxis": "y2",
                    "type": "scatter",
                    "name": "- Enrichment",
                    "y": where(is_, signals, 0),
                    "line": {"width": 0, "color": color},
                    "fill": "tozeroy",
                }
            )

        plot_plotly({"layout": layout, "data": data}, html_file_path)

    return enrichment
