from numpy import absolute, arange, asarray, where

from .plot_plotly_figure import plot_plotly_figure


def compute_set_enrichment(
    element_score,
    set_elements,
    statistic="auc",
    plot=True,
    title="SEA Mountain Plot",
    element_score_name="Element Score",
    annotation_text_font_size=10,
    annotation_text_width=100,
    annotation_text_yshift=50,
    html_file_path=None,
):

    set_element_ = {set_element: None for set_element in set_elements}

    hit = asarray(
        [
            element_score_element in set_element_
            for element_score_element in element_score.index
        ],
        dtype=int,
    )

    up = hit * absolute(element_score.values)

    up /= up.sum()

    down = 1.0 - hit

    down /= down.sum()

    cumsum = (up - down).cumsum()

    gsea_score = cumsum.sum()

    if not plot:

        return gsea_score

    layout = {
        "title": {"text": title, "x": 0.5, "xanchor": "center"},
        "xaxis": {"anchor": "y", "title": "Rank"},
        "yaxis": {"domain": (0, 0.16), "title": element_score_name},
        "yaxis2": {"domain": (0.20, 1), "title": "Enrichment"},
    }

    data = []

    grid = arange(cumsum.size)

    line_width = 3.2

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Cumulative Sum",
            "x": grid,
            "y": cumsum,
            "line": {"width": line_width, "color": "#20d9ba"},
            "fill": "tozeroy",
        }
    )

    peek_index = absolute(cumsum).argmax()

    negative_color = "#4e40d8"

    positive_color = "#ff1968"

    if gsea_score < 0:

        color = negative_color

    else:

        color = positive_color

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Peak ({:.3f})".format(gsea_score),
            "x": (grid[peek_index],),
            "y": (cumsum[peek_index],),
            "mode": "markers",
            "marker": {"size": 12, "color": color},
        }
    )

    in_indices = where(hit)[0]

    element_xs = grid[in_indices]

    element_texts = element_score[in_indices].index

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Element",
            "x": element_xs,
            "y": (0,) * element_xs.size,
            "text": element_texts,
            "mode": "markers",
            "marker": {
                "symbol": "line-ns-open",
                "size": 16,
                "color": "#9017e6",
                "line": {"width": line_width},
            },
            "hoverinfo": "x+text",
        }
    )

    is_negative = element_score < 0

    for indices, name, color in (
        (is_negative, "- Element Score", negative_color),
        (~is_negative, "+ Element Score", positive_color),
    ):

        data.append(
            {
                "type": "scatter",
                "name": name,
                "x": grid[indices],
                "y": element_score[indices],
                "line": {"width": line_width, "color": color},
                "fill": "tozeroy",
            }
        )

    layout["annotations"] = [
        {
            "x": x,
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
        for i, (x, str_) in enumerate(zip(element_xs, element_texts))
    ]

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)

    return gsea_score
