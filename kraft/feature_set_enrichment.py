from multiprocessing import Pool

from numpy import asarray, nan, where
from pandas import DataFrame, Series

from .array import normalize
from .information import get_jsd
from .plot import plot_plotly


def score_sample_and_set(
    element_scores,
    elements,
    sort_element_scores=True,
    method="ks",
    plot_process=False,
    plot=True,
    title="Score Set",
    element_socre_name="Element Score",
    annotation_text_font_size=8,
    annotation_text_width=160,
    annotation_text_yshift=32,
    file_path=None,
):

    if sort_element_scores:

        element_scores = element_scores.sort_values()

    elements = {element: None for element in elements}

    h_1 = asarray(
        tuple(element in elements for element in element_scores.index.to_numpy()),
        dtype=float,
    )

    if h_1.sum() == 0:

        return nan

    m_1 = 1 - h_1

    a = element_scores.abs().to_numpy()

    if method == "ks":

        h_1 *= a

        h_1_p = h_1 / h_1.sum()

        m_1_p = m_1 / m_1.sum()

        if plot_process:

            plot_plotly(
                {
                    "layout": {"title": {"text": "Probability"}},
                    "data": [
                        {"name": "Hit", "y": h_1_p, "mode": "lines"},
                        {"name": "Miss", "y": m_1_p, "mode": "lines"},
                    ],
                },
            )

        h_1_p_rc = h_1_p[::-1].cumsum()[::-1]

        m_1_p_rc = m_1_p[::-1].cumsum()[::-1]

        if plot_process:

            plot_plotly(
                {
                    "layout": {"title": {"text": "< Cumulative Probability"}},
                    "data": [
                        {"name": "Hit", "y": h_1_p_rc, "mode": "lines"},
                        {"name": "Miss", "y": m_1_p_rc, "mode": "lines"},
                    ],
                },
            )

        r_h = h_1_p_rc

        r_m = m_1_p_rc

        r = h_1_p_rc - m_1_p_rc

        s = r

    elif method == "pk":
        print(h_1.sum(), m_1.sum())

        h_a = h_1 * a

        m_a = m_1 * a

        h_a_p = h_a / h_a.sum()

        m_a_p = m_a / m_a.sum()

        a_p = a / a.sum()

        if plot_process:

            plot_plotly(
                {
                    "layout": {"title": {"text": "Magnitude"}},
                    "data": [
                        {"name": "Hit", "y": h_a_p, "mode": "lines"},
                        {"name": "Miss", "y": m_a_p, "mode": "lines"},
                        {"name": "All", "y": a_p, "mode": "lines"},
                    ],
                },
            )

        add = 1e-8

        h_a_p_lc = h_a_p.cumsum() + add

        h_a_p_rc = h_a_p[::-1].cumsum()[::-1] + add

        m_a_p_lc = m_a_p.cumsum() + add

        m_a_p_rc = m_a_p[::-1].cumsum()[::-1] + add

        a_p_lc = a_p.cumsum() + add

        a_p_rc = a_p[::-1].cumsum()[::-1] + add

        if plot_process:

            plot_plotly(
                {
                    "layout": {"title": {"text": "Cumulative Probability >"}},
                    "data": [
                        {"name": "Hit", "y": h_a_p_lc, "mode": "lines"},
                        {"name": "Miss", "y": m_a_p_lc, "mode": "lines"},
                        {"name": "All", "y": a_p_lc, "mode": "lines"},
                    ],
                },
            )

            plot_plotly(
                {
                    "layout": {"title": {"text": "< Cumulative Probability"}},
                    "data": [
                        {"name": "Hit", "y": h_a_p_rc, "mode": "lines"},
                        {"name": "Miss", "y": m_a_p_rc, "mode": "lines"},
                        {"name": "All", "y": a_p_rc, "mode": "lines"},
                    ],
                },
            )

        l_h, l_m, l = get_jsd(h_a_p_lc, m_a_p_lc, reference_vector=a_p_lc)

        r_h, r_m, r = get_jsd(h_a_p_rc, m_a_p_rc, reference_vector=a_p_rc)

        s = r - l

    if plot_process:

        opacity = 0.32

        base = {
            "name": "Signal",
            "mode": "lines",
            "marker": {"color": "#ff1968"},
        }

        if method == "pk":

            plot_plotly(
                {
                    "layout": {"title": {"text": "Signal >"}},
                    "data": [
                        {"name": "Hit", "y": l_h, "opacity": opacity, "mode": "lines"},
                        {"name": "Miss", "y": l_m, "opacity": opacity, "mode": "lines"},
                        {"y": l, **base},
                    ],
                },
            )

        plot_plotly(
            {
                "layout": {"title": {"text": "< Signal"}},
                "data": [
                    {"name": "Hit", "y": r_h, "opacity": opacity, "mode": "lines"},
                    {"name": "Miss", "y": r_m, "opacity": opacity, "mode": "lines"},
                    {"y": r, **base},
                ],
            },
        )

    score = s.sum() / s.size

    if plot:

        y_fraction = 0.16

        layout = {
            "title": {
                "text": "{}<br>Score (method={}) = {:.2f}".format(title, method, score),
                "x": 0.5,
            },
            "xaxis": {"anchor": "y"},
            "yaxis": {"domain": (0, y_fraction), "title": element_socre_name},
            "yaxis2": {"domain": (y_fraction + 0.08, 1)},
            "legend_orientation": "h",
            "legend": {"y": -0.24},
        }

        h_i = where(h_1)[0]

        data = [
            {
                "name": "Element Score ({})".format(element_scores.size),
                "y": element_scores.to_numpy(),
                "text": element_scores.index.to_numpy(),
                "mode": "lines",
                "line": {"width": 0, "color": "#20d8ba"},
                "fill": "tozeroy",
            },
            {
                "name": "Element ({})".format(h_i.size),
                "yaxis": "y2",
                "x": h_i,
                "y": (0,) * h_i.size,
                "text": element_scores.index.to_numpy()[h_i],
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

        for name, _1, color in (
            ("- Enrichment", s < 0, "#0088ff"),
            ("+ Enrichment", 0 < s, "#ff1968"),
        ):

            data.append(
                {
                    "name": name,
                    "yaxis": "y2",
                    "y": where(_1, s, 0),
                    "mode": "lines",
                    "line": {"width": 0, "color": color},
                    "fill": "tozeroy",
                }
            )

        plot_plotly({"layout": layout, "data": data}, file_path=file_path)

    return score


def _score_sample_and_sets(element_scores, set_to_elements, method):

    print(element_scores.name)

    element_scores = Series(
        data=normalize(element_scores.to_numpy(), "-0-"),
        index=element_scores.index,
        name=element_scores.name,
    ).sort_values()

    return tuple(
        score_sample_and_set(
            element_scores,
            elements,
            sort_element_scores=False,
            method=method,
            plot=False,
        )
        for elements in asarray(tuple(set_to_elements.values()))
    )


def score_samples_and_sets(
    element_x_sample, set_to_elements, method="ks", n_job=1, file_path=None
):

    pool = Pool(processes=n_job)

    set_x_sample = DataFrame(
        data=asarray(
            pool.starmap(
                _score_sample_and_sets,
                (
                    (scores, set_to_elements, method)
                    for _, scores in element_x_sample.items()
                ),
            )
        ).T,
        index=set_to_elements.keys(),
        columns=element_x_sample.columns,
    )

    if file_path is not None:

        set_x_sample.to_csv(file_path, sep="\t")

    return set_x_sample
