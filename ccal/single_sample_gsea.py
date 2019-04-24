from numpy import absolute, asarray, nan

from .plot_and_save import plot_and_save


def single_sample_gsea(
    gene_score,
    gene_set_genes,
    statistic="ks",
    plot=True,
    title="GSEA Mountain Plot",
    gene_score_name="Gene Score",
    annotation_text_font_size=16,
    annotation_text_width=88,
    annotation_text_yshift=64,
    html_file_path=None,
):

    gene_score_no_na_sorted = gene_score  # .dropna().sort_values(ascending=False)

    gene_set_genes = {gene_set_gene: None for gene_set_gene in gene_set_genes}

    in_ = asarray(
        [
            gene_score_gene in gene_set_genes
            for gene_score_gene in gene_score_no_na_sorted.index.values
        ],
        dtype=int,
    )

    if not in_.any():

        print("Gene scores did not have any of the gene-set genes.")

        return nan

    gene_score_sorted_values = gene_score_no_na_sorted.values

    hit = absolute(gene_score_sorted_values) * in_

    hit /= hit.sum()

    miss = 1.0 - in_

    miss /= miss.sum()

    y = hit - miss

    cumulative_sums = y.cumsum()

    if statistic == "ks":

        max_ = cumulative_sums.max()

        min_ = cumulative_sums.min()

        if absolute(min_) < absolute(max_):

            score = max_

        else:

            score = min_

    elif statistic == "auc":

        score = cumulative_sums.sum()

    if not plot:

        return score

    layout = {
        "title": {"text": title},
        "xaxis": {"anchor": "y", "title": "Rank"},
        "yaxis": {"domain": (0, 0.16), "title": gene_score_name},
        "yaxis2": {"domain": (0.20, 1), "title": "Enrichment"},
    }

    data = []

    grid = asarray(range(cumulative_sums.size))

    line_width = 3.2

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Cumulative Sum",
            "x": grid,
            "y": cumulative_sums,
            "line": {"width": line_width, "color": "#20d9ba"},
            "fill": "tozeroy",
        }
    )

    cumulative_sums_argmax = absolute(cumulative_sums).argmax()

    negative_color = "#4e40d8"

    positive_color = "#ff1968"

    if score < 0:

        color = negative_color

    else:

        color = positive_color

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Peak ({:.3f})".format(score),
            "x": (grid[cumulative_sums_argmax],),
            "y": (cumulative_sums[cumulative_sums_argmax],),
            "mode": "markers",
            "marker": {"size": 12, "color": color},
        }
    )

    gene_xs = tuple(i for i in grid if in_[i])

    gene_texts = tuple(
        "<b>{}</b>".format(text) for text in gene_score_no_na_sorted[in_].index
    )

    data.append(
        {
            "yaxis": "y2",
            "type": "scatter",
            "name": "Gene",
            "x": gene_xs,
            "y": (0,) * len(gene_xs),
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

    is_negative = gene_score_no_na_sorted < 0

    for indices, name, color in (
        (is_negative, "- Gene Score", negative_color),
        (~is_negative, "+ Gene Score", positive_color),
    ):

        data.append(
            {
                "type": "scatter",
                "name": name,
                "x": grid[indices],
                "y": gene_score_no_na_sorted[indices],
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
            "text": text,
            "showarrow": False,
            "font": {"size": annotation_text_font_size},
            "textangle": -90,
            "width": annotation_text_width,
            "borderpad": 0,
            "yshift": (-annotation_text_yshift, annotation_text_yshift)[i % 2],
        }
        for i, (x, text) in enumerate(zip(gene_xs, gene_texts))
    ]

    plot_and_save({"layout": layout, "data": data}, html_file_path)

    return score
