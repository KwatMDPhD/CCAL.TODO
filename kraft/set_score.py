from numpy import absolute, asarray, where

from .information import get_jsd
from .plot import plot_plotly


def score_set(
    element_value,
    set_elements,
    method="classic",
    symmetry=False,
    plot_=False,
    plot=True,
    title="Score Set",
    element_value_name="Element Value",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    html_file_path=None,
):

    element_value = element_value.sort_values()

    values = element_value.values

    magnitudes = absolute(values)

    set_elements = {element: None for element in set_elements}

    h_is = asarray(
        tuple(element in set_elements for element in element_value.index), dtype=float
    )

    m_is = 1 - h_is

    if method == "classic":

        h_lf, m_lf, h_rf, m_rf = cumulate_rank(magnitudes, h_is, m_is, plot_)

        ls_h, ls_m, ls = h_lf, m_lf, h_lf - m_lf

        rs_h, rs_m, rs = h_rf, m_rf, h_rf - m_rf

    elif method == "2020":

        h_lf, m_lf, r_lf, h_rf, m_rf, r_rf = cumulate_magnitude(
            magnitudes, h_is, m_is, plot_
        )

        ls_h, ls_m, ls = get_jsd(h_lf, m_lf, vector_reference=r_lf)

        rs_h, rs_m, rs = get_jsd(h_rf, m_rf, vector_reference=r_rf)

    if plot_:

        opacity = 0.32

        signal_template = {
            "name": "Signal",
            "mode": "lines",
            "marker": {"color": "#ff1968"},
        }

        plot_plotly(
            {
                "layout": {"title": {"text": "Signal >"}},
                "data": [
                    {"name": "Hit", "y": ls_h, "opacity": opacity, "mode": "lines"},
                    {"name": "Miss", "y": ls_m, "opacity": opacity, "mode": "lines"},
                    {"y": ls, **signal_template},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {"title": {"text": "< Signal"}},
                "data": [
                    {"name": "Hit", "y": rs_h, "opacity": opacity, "mode": "lines"},
                    {"name": "Miss", "y": rs_m, "opacity": opacity, "mode": "lines"},
                    {"y": rs, **signal_template},
                ],
            },
        )

    if symmetry:

        s = rs - ls

    else:

        s = rs

    score = s.sum() / s.size

    if plot:

        y_fraction = 0.16

        layout = {
            "title": {
                "text": "{}<br>Score (method={}, symmetry={}) = {:.2f}".format(
                    title, method, symmetry, score
                ),
                "x": 0.5,
            },
            "xaxis": {"anchor": "y"},
            "yaxis": {"domain": (0, y_fraction), "title": element_value_name},
            "yaxis2": {"domain": (y_fraction + 0.08, 1)},
            "legend_orientation": "h",
            "legend": {"y": -0.24},
        }

        h_i = where(h_is)[0]

        data = [
            {
                "name": "Element Value ({})".format(values.size),
                "y": values,
                "text": element_value.index,
                "mode": "lines",
                "line": {"width": 0, "color": "#20d8ba"},
                "fill": "tozeroy",
            },
            {
                "name": "Element ({})".format(h_i.size),
                "yaxis": "y2",
                "x": h_i,
                "y": (0,) * h_i.size,
                "text": element_value.index[h_i],
                "mode": "markers",
                "marker": {
                    "symbol": "line-ns-open",
                    "size": 8,
                    "color": "#2e211b",
                    "line": {"width": 1.6},
                },
                "hoverinfo": "x+text",
            },
        ]

        for name, is_, color in (
            ("- Enrichment", s < 0, "#0088ff"),
            ("+ Enrichment", 0 < s, "#ff1968"),
        ):

            data.append(
                {
                    "name": name,
                    "yaxis": "y2",
                    "y": where(is_, s, 0),
                    "mode": "lines",
                    "line": {"width": 0, "color": color},
                    "fill": "tozeroy",
                }
            )

        plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)

    return score


def get_c(vector):

    lc = vector.cumsum()

    rc = vector[::-1].cumsum()[::-1]

    add = 1e-8

    return lc + add, rc + add


def cumulate_rank(magnitudes, h_is, m_is, plot_):

    h_is *= magnitudes

    h_r_p = h_is / h_is.sum()

    m_r_p = m_is / m_is.sum()

    if plot_:

        plot_plotly(
            {
                "layout": {"title": {"text": "PDF"}},
                "data": [
                    {"name": "Hit", "y": h_r_p, "mode": "lines"},
                    {"name": "Miss", "y": m_r_p, "mode": "lines"},
                ],
            },
        )

    h_r_lc, h_r_rc = get_c(h_r_p)

    m_r_lc, m_r_rc = get_c(m_r_p)

    if plot_:

        plot_plotly(
            {
                "layout": {"title": {"text": "CDF >"}},
                "data": [
                    {"name": "Hit", "y": h_r_lc, "mode": "lines"},
                    {"name": "Miss", "y": m_r_lc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {"title": {"text": "< CDF"}},
                "data": [
                    {"name": "Hit", "y": h_r_rc, "mode": "lines"},
                    {"name": "Miss", "y": m_r_rc, "mode": "lines"},
                ],
            },
        )

    return h_r_lc, m_r_lc, h_r_rc, m_r_rc


def cumulate_magnitude(magnitudes, h_is, m_is, plot_):

    v = magnitudes

    h_v = v * h_is

    m_v = v * m_is

    v /= v.sum()

    h_v /= h_v.sum()

    m_v /= m_v.sum()

    if plot_:

        plot_plotly(
            {
                "layout": {"title": {"text": "Magnitude"}},
                "data": [
                    {"name": "Hit", "y": h_v, "mode": "lines"},
                    {"name": "Miss", "y": m_v, "mode": "lines"},
                    {"name": "All", "y": v, "mode": "lines"},
                ],
            },
        )

    h_v_lc, h_v_rc = get_c(h_v)

    m_v_lc, m_v_rc = get_c(m_v)

    v_lc, v_rc = get_c(v)

    if plot_:

        plot_plotly(
            {
                "layout": {"title": {"text": "CDF >"}},
                "data": [
                    {"name": "Hit", "y": h_v_lc, "mode": "lines"},
                    {"name": "Miss", "y": m_v_lc, "mode": "lines"},
                    {"name": "All", "y": v_lc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {"title": {"text": "< CDF"}},
                "data": [
                    {"name": "Hit", "y": h_v_rc, "mode": "lines"},
                    {"name": "Miss", "y": m_v_rc, "mode": "lines"},
                    {"name": "All", "y": v_rc, "mode": "lines"},
                ],
            },
        )

    return h_v_lc, m_v_lc, v_lc, h_v_rc, m_v_rc, v_rc
