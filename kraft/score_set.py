from numpy import absolute, asarray, where

from .get_jsd import get_jsd
from .get_zd import get_zd
from .plot_plotly import plot_plotly


def score_set(
    element_value,
    set_elements,
    method=("rank", "cdf", "s0", "area"),
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

    set_elements = {element: None for element in set_elements}

    h_is = asarray(
        tuple(element in set_elements for element in element_value.index), dtype=float
    )

    m_is = 1 - h_is

    h_i = where(h_is)[0]

    m_i = where(m_is)[0]

    opacity = 0.32

    signal_template = {
        "name": "Signal",
        "mode": "lines",
        "marker": {"color": "#ff1968"},
    }

    if method[1] == "pdf":

        if method[0] == "rank":

            h_f, m_f, r_f = get_r_p(values, h_is, m_is, plot_)

        elif method[0] == "value":

            h_f, m_f, r_f = get_v_p(values, h_is, m_is, plot_)

        if method[2] == "s0":

            s_h, s_m, s = h_f, m_f, h_f - m_f

        elif method[2] == "s1":

            s_h, s_m, s = get_jsd(h_f, m_f, r_f)

        elif method[2] == "s2":

            s_h, s_m, s = get_zd(h_f, m_f)

    elif method[1] == "cdf":

        if method[0] == "rank":

            h_lf, m_lf, r_lf, h_rf, m_rf, r_rf = get_r_c(values, h_is, m_is, plot_)

        elif method[0] == "value":

            h_lf, m_lf, r_lf, h_rf, m_rf, r_rf = get_v_c(values, h_is, m_is, plot_)

        if method[2] == "s0":

            ls_h, ls_m, ls = h_lf, m_lf, h_lf - m_lf

            rs_h, rs_m, rs = h_rf, m_rf, h_rf - m_rf

        elif method[2] == "s1":

            ls_h, ls_m, ls = get_jsd(h_lf, m_lf, r_lf)

            rs_h, rs_m, rs = get_jsd(h_rf, m_rf, r_rf)

        elif method[2] == "s2":

            ls_h, ls_m, ls = get_zd(h_lf, m_lf)

            rs_h, rs_m, rs = get_zd(h_rf, m_rf)

        if plot_:

            plot_plotly(
                {
                    "layout": {"title": {"text": "Signal >"}},
                    "data": [
                        {
                            "name": "Hit",
                            "y": ls_h,
                            "opacity": opacity,
                            "mode": "lines",
                        },
                        {
                            "name": "Miss",
                            "y": ls_m,
                            "opacity": opacity,
                            "mode": "lines",
                        },
                        {"y": ls, **signal_template},
                    ],
                },
            )

            plot_plotly(
                {
                    "layout": {"title": {"text": "< Signal"}},
                    "data": [
                        {
                            "name": "Hit",
                            "y": rs_h,
                            "opacity": opacity,
                            "mode": "lines",
                        },
                        {
                            "name": "Miss",
                            "y": rs_m,
                            "opacity": opacity,
                            "mode": "lines",
                        },
                        {"y": rs, **signal_template},
                    ],
                },
            )

        if method[:3] == ("rank", "cdf", "s0"):

            s = rs

        else:

            s = rs - ls

    if method[3] == "supreme":

        score = s[absolute(s).argmax()]

    elif method[3] == "area":

        score = s.sum() / s.size

    if plot:

        y_fraction = 0.16

        layout = {
            "title": {
                "text": "{}<br>{} {} {} {} = {:.2f}".format(title, *method, score),
                "x": 0.5,
            },
            "xaxis": {"anchor": "y"},
            "yaxis": {"domain": (0, y_fraction), "title": element_value_name},
            "yaxis2": {"domain": (y_fraction + 0.08, 1)},
            "legend_orientation": "h",
            "legend": {"y": -0.24},
        }

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

        plot_plotly({"layout": layout, "data": data}, html_file_path)

    return score


def get_c(vector):

    lc = vector.cumsum()

    rc = vector[::-1].cumsum()[::-1]

    add = 1e-8

    return lc + add, rc + add


def get_r_p(values, h_is, m_is, plot_):

    h_is *= absolute(values)

    h_r_p = h_is / h_is.sum()

    m_r_p = m_is / m_is.sum()

    r_p = (h_r_p + m_r_p) / 2

    if plot_:

        plot_plotly(
            {
                "layout": {"title": {"text": "PDF"}},
                "data": [
                    {"name": "Hit", "y": h_r_p, "mode": "lines"},
                    {"name": "Miss", "y": m_r_p, "mode": "lines"},
                    {"name": "Center", "y": r_p, "mode": "lines"},
                ],
            },
        )

    return h_r_p, m_r_p, r_p


def get_r_c(values, h_is, m_is, plot_):

    h_r_p, m_r_p, r_p = get_r_p(values, h_is, m_is, plot_)

    h_r_lc, h_r_rc = get_c(h_r_p)

    m_r_lc, m_r_rc = get_c(m_r_p)

    r_lc, r_rc = get_c(r_p)

    if plot_:

        plot_plotly(
            {
                "layout": {"title": {"text": "CDF >"}},
                "data": [
                    {"name": "Hit", "y": h_r_lc, "mode": "lines"},
                    {"name": "Miss", "y": m_r_lc, "mode": "lines"},
                    {"name": "Center", "y": r_lc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {"title": {"text": "< CDF"}},
                "data": [
                    {"name": "Hit", "y": h_r_rc, "mode": "lines"},
                    {"name": "Miss", "y": m_r_rc, "mode": "lines"},
                    {"name": "Center", "y": r_rc, "mode": "lines"},
                ],
            },
        )

    return h_r_lc, m_r_lc, r_lc, h_r_rc, m_r_rc, r_rc


def get_v_p(values, h_is, m_is, plot_):

    v = absolute(values)

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

    return h_v, m_v, v


def get_v_c(values, h_is, m_is, plot_):

    h_v, m_v, v = get_v_p(values, h_is, m_is, plot_)

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
