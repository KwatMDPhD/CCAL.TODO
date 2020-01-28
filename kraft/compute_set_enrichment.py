from numpy import absolute, asarray, isnan, log, where

from .compute_bandwidth import compute_bandwidth
from .estimate_kernel_density import estimate_kernel_density
from .make_grid import make_grid
from .normalize import normalize
from .plot_plotly import plot_plotly


def compute_set_enrichment(
    element_score,
    set_elements,
    method="rank cdf ks",
    power=0,
    n_grid=None,
    plot_analysis=False,
    plot=False,
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
        dtype=float,
    )
    print(r_h)

    r_m = 1 - r_h

    if plot_analysis:

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
                        "name": "Miss ({:.3f})".format(r_m.sum() / r_m.size),
                        "x": r_m_i,
                        "y": element_score.values[r_m_i],
                    },
                    {
                        "name": "Hit ({:.3f})".format(r_h.sum() / r_h.size),
                        "x": r_h_i,
                        "y": element_score.values[r_h_i],
                    },
                ],
            },
        )

    def compute_k1(h, m, c):

        klhc = h * log(h / c)

        klhc[isnan(klhc)] = 0

        klmc = m * log(m / c)

        klmc[isnan(klmc)] = 0

        return klhc, klmc, klhc - klmc

    def compute_k2(h, m):

        klhm = h * log(h / m)

        klhm[isnan(klhm)] = 0

        klmh = m * log(m / h)

        klmh[isnan(klmh)] = 0

        return klhm, klmh, klhm - klmh

    def get_c(p):

        return normalize(p[::-1].cumsum()[::-1], "0-1")

    if method.startswith("rank cdf"):

        if power != 0:

            r_h *= absolute(element_score.values) ** power

        r_h_p = r_h / r_h.sum()

        r_m_p = r_m / r_m.sum()

        if plot_analysis:

            plot_plotly(
                {
                    "layout": {
                        "title": {"text": "PDF(rank | event)"},
                        "xaxis": {"title": {"text": "Rank"}},
                        "yaxis": {"title": {"text": "Probability"}},
                    },
                    "data": [
                        {"name": "Miss", "y": r_m_p},
                        {"name": "Hit", "y": r_h_p},
                    ],
                },
            )

        r_h_c = get_c(r_h_p)

        r_m_c = get_c(r_m_p)

        r_c = (r_h_c + r_m_c) / 2

        if plot_analysis:

            plot_plotly(
                {
                    "layout": {
                        "title": {"text": "CDF(rank | event)"},
                        "xaxis": {"title": {"text": "Rank"}},
                        "yaxis": {"title": {"text": "Cumulative Probability"}},
                    },
                    "data": [
                        {"name": "Miss", "y": r_m_c},
                        {"name": "Hit", "y": r_h_c},
                        {"name": "Center", "y": r_c},
                    ],
                },
            )

        if method.endswith("ks"):

            h, m, s = None, None, r_h_c - r_m_c

            enrichment = s[absolute(s).argmax()]

        else:

            if method.endswith("k1"):

                h, m, s = compute_k1(r_h_c, r_m_c, r_c)

            elif method.endswith("k2"):

                h, m, s = compute_k2(r_h_c, r_m_c)

            enrichment = s.sum() / n_grid

    if method.startswith("score"):

        s_b = compute_bandwidth(element_score.values)

        if n_grid is None:

            n_grid = element_score.size * 3

        s_g = make_grid(
            element_score.values.min(), element_score.values.max(), 1 / 3, n_grid
        )

        dg = s_g[1] - s_g[0]

        def get_p(vector):

            point_x_dimension, kernel_densities = estimate_kernel_density(
                vector.reshape(vector.size, 1),
                bandwidths=(s_b,),
                grids=(s_g,),
                plot=False,
            )

            return kernel_densities / (kernel_densities.sum() * dg)

        s_h_p = get_p(element_score.values[where(r_h)])

        s_m_p = get_p(element_score.values[where(r_m)])

        s_p = get_p(element_score.values)

        if plot_analysis:

            plot_plotly(
                {
                    "layout": {
                        "title": {"text": "PDF(score | event)"},
                        "xaxis": {"title": {"text": "Score"}},
                        "yaxis": {"title": {"text": "Probability"}},
                    },
                    "data": [
                        {"name": "Miss", "x": s_g, "y": s_m_p},
                        {"name": "Hit", "x": s_g, "y": s_h_p},
                        {"name": "All", "x": s_g, "y": s_p},
                    ],
                },
            )

        if "pdf" in method:

            if method.endswith("k1"):

                h, m, s = compute_k1(s_h_p, s_m_p, s_p)

            elif method.endswith("k2"):

                h, m, s = compute_k2(s_h_p, s_m_p)

            enrichment = s.sum() / n_grid

        elif "cdf" in method:

            s_h_c = get_c(s_h_p)

            s_m_c = get_c(s_m_p)

            s_c = get_c(s_p)

            if plot_analysis:

                plot_plotly(
                    {
                        "layout": {
                            "title": {"text": "CDF(score | event)"},
                            "xaxis": {"title": {"text": "Score"}},
                            "yaxis": {"title": {"text": "Cumulative Probability"}},
                        },
                        "data": [
                            {"name": "Miss", "x": s_g, "y": s_m_c},
                            {"name": "Hit", "x": s_g, "y": s_h_c},
                            {"name": "All", "x": s_g, "y": s_c},
                        ],
                    },
                )

            if method.endswith("ks"):

                h, m, s = None, None, s_h_c - s_m_c

                enrichment = s[absolute(s).argmax()]

            else:

                if method.endswith("k1"):

                    h, m, s = compute_k1(s_h_c, s_m_c, s_c)

                elif method.endswith("k2"):

                    h, m, s = compute_k2(s_h_c, s_m_c)

                enrichment = s.sum() / n_grid

    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {
                        "text": "method: {}<br>score: {:.3f}".format(method, enrichment)
                    }
                },
                "data": [
                    {"name": "Miss", "y": m, "opacity": 0.32},
                    {"name": "Hit", "y": h, "opacity": 0.32},
                    {"name": "Signal", "y": s},
                ],
            },
        )

    return enrichment
