from numpy import absolute, asarray, where


def compute_set_enrichment(
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

    assert 1 < r_h.sum()

    r_m = 1 - r_h

    assert 1 < r_m.sum()

    #
    r_h_i = where(r_h)[0]

    #
    r_h_v = r_h * absolute(element_score.values)

    if power != 0:

        r_h_v **= power

    r_h_p = r_h_v / r_h_v.sum()

    r_h_c = r_h_p[::-1].cumsum()[::-1]

    #
    r_m_v = r_m

    r_m_p = r_m_v / r_m_v.sum()

    r_m_c = r_m_p[::-1].cumsum()[::-1]

    #
    str_signals = {}

    #
    str_signals["rank cdf ks"] = r_h_c - r_m_c

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
    plot({"layout": layout, "data": data}, html_file_path)

    return enrichment
