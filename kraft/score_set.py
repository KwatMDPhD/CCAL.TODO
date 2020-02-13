from numpy import absolute, asarray, isnan, log, where

from .estimate_density import estimate_density
from .get_bandwidth import get_bandwidth
from .make_grid import make_grid
from .normalize import normalize
from .plot_plotly import plot_plotly


def score_set(
    element_score,
    set_elements,
    method=("rank", "cdf", "s0", "supreme"),
    power=1,
    plot_=False,
    plot=True,
    title="Set Enrichment",
    element_score_name="Element Score",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    html_file_path=None,
):

    element_score = element_score.sort_values()

    element_score_ = element_score.values

    set_element_ = {element: None for element in set_elements}

    r_h = asarray(
        tuple(element in set_element_ for element in element_score.index), dtype=float
    )

    r_m = 1 - r_h

    if plot_:

        r_h_i = where(r_h)[0]

        r_m_i = where(r_m)[0]

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Element Score"},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Score"}},
                },
                "data": [
                    {
                        "name": "Miss ({:.1%})".format(r_m.sum() / r_m.size),
                        "x": r_m_i,
                        "y": element_score_[r_m_i],
                        "mode": "markers",
                    },
                    {
                        "name": "Hit ({:.1%})".format(r_h.sum() / r_h.size),
                        "x": r_h_i,
                        "y": element_score_[r_h_i],
                        "mode": "markers",
                    },
                ],
            },
        )

    def get_c(p):

        return normalize(p[::-1].cumsum()[::-1], "0-1")

    def get_s0(h, m):

        return h - m

    def get_s1(h, m, c):

        kl_hc = h * log(h / c)

        kl_hc[isnan(kl_hc)] = 0

        kl_mc = m * log(m / c)

        kl_mc[isnan(kl_mc)] = 0

        return kl_hc, kl_mc, kl_hc - kl_mc

    def get_s2(h, m):

        kl_hm = h * log(h / m)

        kl_hm[isnan(kl_hm)] = 0

        kl_mh = m * log(m / h)

        kl_mh[isnan(kl_mh)] = 0

        return kl_hm, kl_mh, kl_hm - kl_mh

    if method[0] == "rank":

        if power != 0:

            r_h *= absolute(element_score_) ** power

        r_h_p = r_h / r_h.sum()

        r_m_p = r_m / r_m.sum()

        r_p = (r_h_p + r_m_p) / 2

        if plot_:

            plot_plotly(
                {
                    "layout": {
                        "title": {"text": "PDF(rank | event)"},
                        "xaxis": {"title": {"text": "Rank"}},
                        "yaxis": {"title": {"text": "Probability"}},
                    },
                    "data": [
                        {"name": "Miss", "y": r_m_p, "mode": "lines"},
                        {"name": "Hit", "y": r_h_p, "mode": "lines"},
                        {"name": "Center", "y": r_p, "mode": "lines"},
                    ],
                },
            )

        if method[1] == "pdf":

            h = r_h_p

            m = r_m_p

            c = r_p

        if method[1] == "cdf":

            r_h_c = get_c(r_h_p)

            r_m_c = get_c(r_m_p)

            r_c = (r_h_c + r_m_c) / 2

            if plot_:

                plot_plotly(
                    {
                        "layout": {
                            "title": {"text": "CDF(rank | event)"},
                            "xaxis": {"title": {"text": "Rank"}},
                            "yaxis": {"title": {"text": "Cumulative Probability"}},
                        },
                        "data": [
                            {"name": "Miss", "y": r_m_c, "mode": "lines"},
                            {"name": "Hit", "y": r_h_c, "mode": "lines"},
                            {"name": "Center", "y": r_c, "mode": "lines"},
                        ],
                    },
                )

            h = r_h_c

            m = r_m_c

            c = r_c

    elif method[0] == "score":

        s_b = get_bandwidth(element_score_)

        s_g = make_grid(
            element_score_.min(), element_score_.max(), 1 / 3, element_score_.size * 3,
        )

        dg = s_g[1] - s_g[0]

        def get_p(vector):

            density = estimate_density(
                vector.reshape(vector.size, 1),
                bandwidths=(s_b,),
                grids=(s_g,),
                plot=False,
            )[1]

            return density / (density.sum() * dg)

        s_h_p = get_p(element_score_[where(r_h)])

        s_m_p = get_p(element_score_[where(r_m)])

        s_p = get_p(element_score_)

        if plot_:

            plot_plotly(
                {
                    "layout": {
                        "title": {"text": "PDF(score | event)"},
                        "xaxis": {"title": {"text": "Score"}},
                        "yaxis": {"title": {"text": "Probability"}},
                    },
                    "data": [
                        {"name": "Miss", "x": s_g, "y": s_m_p, "mode": "lines"},
                        {"name": "Hit", "x": s_g, "y": s_h_p, "mode": "lines"},
                        {"name": "All", "x": s_g, "y": s_p, "mode": "lines"},
                    ],
                },
            )

        if method[1] == "pdf":

            h = s_h_p

            m = s_m_p

            c = s_p

        elif "cdf" in method:

            s_h_c = get_c(s_h_p)

            s_m_c = get_c(s_m_p)

            s_c = get_c(s_p)

            if plot_:

                plot_plotly(
                    {
                        "layout": {
                            "title": {"text": "CDF(score | event)"},
                            "xaxis": {"title": {"text": "Score"}},
                            "yaxis": {"title": {"text": "Cumulative Probability"}},
                        },
                        "data": [
                            {"name": "Miss", "x": s_g, "y": s_m_c, "mode": "lines"},
                            {"name": "Hit", "x": s_g, "y": s_h_c, "mode": "lines"},
                            {"name": "All", "x": s_g, "y": s_c, "mode": "lines"},
                        ],
                    },
                )

            h = s_h_c

            m = s_m_c

            c = s_c

    if method[2] == "s0":

        s = get_s0(h, m)

    elif method[2] == "s1":

        h, m, s = get_s1(h, m, c)

    elif method[2] == "s2":

        h, m, s = get_s2(h, m)

    if method[0] == "score":

        index = [absolute(score_ - s_g).argmin() for score_ in element_score_]

        h = h[index]

        m = m[index]

        s = s[index]

    if method[3] == "supreme":

        score = s[absolute(s).argmax()]

    elif method[3] == "area":

        score = s.sum()

    if plot:

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
                    {"name": "Miss", "y": m, "opacity": 0.32, "mode": "lines"},
                    {"name": "Hit", "y": h, "opacity": 0.32, "mode": "lines"},
                    {"name": "Signal", "y": s, "mode": "lines"},
                ],
            },
        )

    return score
