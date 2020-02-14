from numpy import absolute, asarray, where

from .accumulate import accumulate
from .estimate_density import estimate_density
from .get_bandwidth import get_bandwidth
from .get_s1 import get_s1
from .get_s2 import get_s2
from .make_grid import make_grid
from .plot_plotly import plot_plotly


def score_set(
    element_value,
    set_elements,
    method=("rank", "cdf", "s0", "supreme"),
    power=1,
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

    set_element_ = {element: None for element in set_elements}

    h_is = asarray(
        tuple(element in set_element_ for element in element_value.index), dtype=float
    )

    m_is = 1 - h_is

    h_i = where(h_is)[0]

    m_i = where(m_is)[0]

    h_r = h_is * absolute(values) ** power

    m_r = m_is

    h_r_p = h_r / h_r.sum()

    m_r_p = m_r / m_r.sum()

    r_p = (h_r_p + m_r_p) / 2

    h_r_lc, h_r_rc = accumulate(h_r_p)

    m_r_lc, m_r_rc = accumulate(m_r_p)

    r_lc = (h_r_lc + m_r_lc) / 2

    r_rc = (h_r_rc + m_r_rc) / 2

    b = get_bandwidth(values)

    g = make_grid(values.min(), values.max(), 1 / 3, values.size * 3)

    dg = g[1] - g[0]

    def get_p(vector):

        density = estimate_density(
            vector.reshape(vector.size, 1), bandwidths=(b,), grids=(g,), plot=False
        )[1]

        return density / (density.sum() * dg)

    h_v_p = get_p(values[h_i])

    m_v_p = get_p(values[m_i])

    v_p = get_p(values)

    h_v_lc, h_v_rc = accumulate(h_v_p)

    m_v_lc, m_v_rc = accumulate(m_v_p)

    v_lc, v_rc = accumulate(v_p)

    if method[0] == "rank":

        if method[1] == "pdf":

            h = h_r_p

            m = m_r_p

            r = r_p

        elif method[1] == "cdf":

            h = h_r_rc

            m = m_r_rc

            r = r_rc

    elif method[0] == "value":

        if method[1] == "pdf":

            h = h_v_p

            m = m_v_p

            r = v_p

        elif method[1] == "cdf":

            h = h_v_rc

            m = m_v_rc

            r = v_rc

    if method[2] == "s0":

        s_h, s_m, s = h, m, h - m

    elif method[2] == "s1":

        s_h, s_m, s = get_s1(h, m, r)

    elif method[2] == "s2":

        s_h, s_m, s = get_s2(h, m)

    if method[0] == "value":

        index = [absolute(value - g).argmin() for value in values]

        s_h = s_h[index]

        s_m = s_m[index]

        s = s[index]

    if method[3] == "supreme":

        score = s[absolute(s).argmax()]

    elif method[3] == "area":

        score = s.sum()

    if plot_:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Element Value"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Value"}},
                },
                "data": [
                    {
                        "name": "Miss ({:.1%})".format(m_is.sum() / m_is.size),
                        "x": m_i,
                        "y": values[m_i],
                        "mode": "markers",
                    },
                    {
                        "name": "Hit ({:.1%})".format(h_is.sum() / h_is.size),
                        "x": h_i,
                        "y": values[h_i],
                        "mode": "markers",
                    },
                ],
            },
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "PDF(rank | event)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {"name": "Miss", "y": m_r_p, "mode": "lines"},
                    {"name": "Hit", "y": h_r_p, "mode": "lines"},
                    {"name": "Center", "y": r_p, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "LCDF(rank | event)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {"name": "Miss", "y": m_r_lc, "mode": "lines"},
                    {"name": "Hit", "y": h_r_lc, "mode": "lines"},
                    {"name": "Center", "y": r_lc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "RCDF(rank | event)"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {"name": "Miss", "y": m_r_rc, "mode": "lines"},
                    {"name": "Hit", "y": h_r_rc, "mode": "lines"},
                    {"name": "Center", "y": r_rc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "PDF(value | event)"},
                    "xaxis": {"title": {"text": "Value"}},
                    "yaxis": {"title": {"text": "Probability"}},
                },
                "data": [
                    {"name": "Miss", "x": g, "y": m_v_p, "mode": "lines"},
                    {"name": "Hit", "x": g, "y": h_v_p, "mode": "lines"},
                    {"name": "All", "x": g, "y": v_p, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "LCDF(value | event)"},
                    "xaxis": {"title": {"text": "Value"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {"name": "Miss", "x": g, "y": m_v_lc, "mode": "lines"},
                    {"name": "Hit", "x": g, "y": h_v_lc, "mode": "lines"},
                    {"name": "All", "x": g, "y": v_lc, "mode": "lines"},
                ],
            },
        )

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "RCDF(value | event)"},
                    "xaxis": {"title": {"text": "Value"}},
                    "yaxis": {"title": {"text": "Cumulative Probability"}},
                },
                "data": [
                    {"name": "Miss", "x": g, "y": m_v_rc, "mode": "lines"},
                    {"name": "Hit", "x": g, "y": h_v_rc, "mode": "lines"},
                    {"name": "All", "x": g, "y": v_rc, "mode": "lines"},
                ],
            },
        )

    if plot:

        hm_opacity = 0.24

        plot_plotly(
            {
                "layout": {
                    "title": {
                        "x": 0.5,
                        "text": "{}<br>{} {} {} {} = {:.2f}".format(
                            title, *method, score
                        ),
                    }
                },
                "data": [
                    {"name": "Miss", "y": s_m, "opacity": hm_opacity, "mode": "lines"},
                    {"name": "Hit", "y": s_h, "opacity": hm_opacity, "mode": "lines"},
                    {"name": "Signal", "y": s, "mode": "lines"},
                ],
            },
        )

    return score
