from numpy import absolute, asarray, where

from .estimate_density import estimate_density
from .get_bandwidth import get_bandwidth
from .get_s1 import get_s1
from .get_s2 import get_s2
from .make_grid import make_grid

# from .normalize import normalize
from .plot_plotly import plot_plotly


def score_set(
    element_value,
    set_elements,
    method=("rank", "cdf", "s0", "supreme"),
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

    # if plot_:

    #     plot_plotly(
    #         {
    #             "layout": {"title": {"text": "Element Value"}},
    #             "data": [
    #                 {
    #                     "name": "Hit ({:.1%})".format(h_i.size / values.size),
    #                     "x": h_i,
    #                     "y": values[h_i],
    #                     "mode": "markers",
    #                 },
    #                 {
    #                     "name": "Miss ({:.1%})".format(m_i.size / values.size),
    #                     "x": m_i,
    #                     "y": values[m_i],
    #                     "mode": "lines",
    #                 },
    #             ],
    #         },
    #     )

    opacity = 0.24

    signal_template = {
        "name": "Signal",
        "mode": "lines",
        "marker": {"color": "#ff1968"},
    }

    if method[1] == "pdf":

        if method[0] == "rank":

            h_f, m_f, r_f = get_r_p(values, h_is, m_is, plot_)

        elif method[0] == "value":

            h_f, m_f, r_f = get_v_p_0(values, h_i, m_i, plot_)

        elif method[0] == "value_":

            h_f, m_f, r_f = get_v_p_1(values, h_is, m_is, plot_)

        if method[2] == "s0":

            s_h, s_m, s = h_f, m_f, h_f - m_f

        elif method[2] == "s1":

            s_h, s_m, s = get_s1(h_f, m_f, r_f)

        elif method[2] == "s2":

            s_h, s_m, s = get_s2(h_f, m_f)

    elif method[1] == "cdf":

        if method[0] == "rank":

            h_lf, m_lf, r_lf, h_rf, m_rf, r_rf = get_r_c(values, h_is, m_is, plot_)

        elif method[0] == "value":

            h_lf, m_lf, r_lf, h_rf, m_rf, r_rf = get_v_c_0(values, h_i, m_i, plot_)

        elif method[0] == "value_":

            h_lf, m_lf, r_lf, h_rf, m_rf, r_rf = get_v_c_1(values, h_is, m_is, plot_)

        if method[2] == "s0":

            ls_h, ls_m, ls = h_lf, m_lf, h_lf - m_lf

            rs_h, rs_m, rs = h_rf, m_rf, h_rf - m_rf

        elif method[2] == "s1":

            ls_h, ls_m, ls = get_s1(h_lf, m_lf, r_lf)

            rs_h, rs_m, rs = get_s1(h_rf, m_rf, r_rf)

        elif method[2] == "s2":

            ls_h, ls_m, ls = get_s2(h_lf, m_lf)

            rs_h, rs_m, rs = get_s2(h_rf, m_rf)

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

        s = rs - ls

    if method[3] == "supreme":

        score = s[absolute(s).argmax()]

    elif method[3] == "area":

        score = s.sum() / s.size

    if plot:

        data = [{"y": s, **signal_template}]

        if method[1] == "pdf":

            data = [
                {"name": "Hit", "y": s_h, "opacity": opacity, "mode": "lines"},
                {"name": "Miss", "y": s_m, "opacity": opacity, "mode": "lines"},
            ] + data

        plot_plotly(
            {
                "layout": {
                    "title": {
                        "x": 0.5,
                        "text": "{}<br>{} {} {} {} = {:.2f}".format(
                            title, *method, score
                        ),
                    },
                },
                "data": data,
            },
        )

    return score


def get_c(vector):

    # TODO: fix boundary

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


def get_v_p_0(values, h_i, m_i, plot_):

    b = get_bandwidth(values)

    g = make_grid(values.min(), values.max(), 1 / 3, values.size * 3)

    dg = g[1] - g[0]

    def get_p(vector):

        density = estimate_density(
            vector.reshape(vector.size, 1), bandwidths=(b,), grids=(g,), plot=False
        )[1]

        return density / (density.sum() * dg)

    index = [absolute(value - g).argmin() for value in values]

    h_v_p = get_p(values[h_i])[index]

    m_v_p = get_p(values[m_i])[index]

    v_p = get_p(values)[index]

    if plot_:

        plot_plotly(
            {
                "layout": {"title": {"text": "PDF"}},
                "data": [
                    {"name": "Hit", "y": h_v_p, "mode": "lines"},
                    {"name": "Miss", "y": m_v_p, "mode": "lines"},
                    {"name": "All", "y": v_p, "mode": "lines"},
                ],
            },
        )

    return h_v_p, m_v_p, v_p


def get_v_c_0(values, h_i, m_i, plot_):

    h_v_p, m_v_p, v_p = get_v_p_0(values, h_i, m_i, plot_)

    h_v_lc, h_v_rc = get_c(h_v_p / h_v_p.sum())

    m_v_lc, m_v_rc = get_c(m_v_p / m_v_p.sum())

    v_lc, v_rc = get_c(v_p / v_p.sum())

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


def get_v_p_1(values, h_is, m_is, plot_):

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


def get_v_c_1(values, h_is, m_is, plot_):

    h_v, m_v, v = get_v_p_1(values, h_is, m_is, plot_)

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
