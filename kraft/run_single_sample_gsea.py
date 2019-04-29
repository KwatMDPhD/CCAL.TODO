from numpy import absolute, arange, asarray, where

from .plot_and_save import plot_and_save


def run_single_sample_gsea(
    gene_score,
    gene_set_genes,
    statistic="ks",
    plot=True,
    title="GSEA Mountain Plot",
    gene_score_name="Gene Score",
    annotation_text_font_size=10,
    annotation_text_width=100,
    annotation_text_yshift=50,
    html_file_path=None,
):

    gene_set_gene_None = {gene_set_gene: None for gene_set_gene in gene_set_genes}

    in_ = asarray(
        [
            gene_score_gene in gene_set_gene_None
            for gene_score_gene in gene_score.index.values
        ],
        dtype=int,
    )

    up = in_ * absolute(gene_score.values)

    up /= up.sum()

    down = 1.0 - in_

    down /= down.sum()

    cumsum = (up - down).cumsum()

    if statistic == "ks":

        max_ = cumsum.max()

        min_ = cumsum.min()

        if absolute(min_) < absolute(max_):

            gsea_score = max_

        else:

            gsea_score = min_

    elif statistic == "auc":

        gsea_score = cumsum.sum()

    if not plot:

        return gsea_score

    layout = {
        "title": {"text": title},
        "xaxis": {"anchor": "y", "title": "Rank"},
        "yaxis": {"domain": (0, 0.16), "title": gene_score_name},
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

    in_indices = where(in_)[0]

    gene_xs = grid[in_indices]

    gene_texts = gene_score[in_indices].index

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Gene",
            "x": gene_xs,
            "y": (0,) * gene_xs.size,
            "text": gene_texts,
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

    is_negative = gene_score < 0

    for indices, name, color in (
        (is_negative, "- Gene Score", negative_color),
        (~is_negative, "+ Gene Score", positive_color),
    ):

        data.append(
            {
                "type": "scatter",
                "name": name,
                "x": grid[indices],
                "y": gene_score[indices],
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
        for i, (x, str_) in enumerate(zip(gene_xs, gene_texts))
    ]

    plot_and_save({"layout": layout, "data": data}, html_file_path)

    return gsea_score
