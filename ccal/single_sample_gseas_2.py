from numpy import apply_along_axis, isin, absolute, where


def single_sample_gseas_2(gene_score, gene_sets, statistic="ks"):

    # gene_score = gene_score.dropna().sort_values(ascending=False)

    gene_score__values = gene_score.values

    gene_score__index_values = gene_score.index.values

    gene_sets__values = gene_sets.values

    gene_set_x_in = apply_along_axis(
        lambda gene_set_genes: isin(
            gene_score__index_values, gene_set_genes, assume_unique=True
        ),
        1,
        gene_sets__values,
    ).astype(int)

    hit = absolute(gene_score__values) * gene_set_x_in

    hit = (hit.T / hit.sum(axis=1)).T

    miss = 1 - gene_set_x_in

    miss = (miss.T / miss.sum(axis=1)).T

    y = hit - miss

    y_cumsum = y.cumsum(axis=1)

    if statistic == "ks":

        y_cumsum_max = y_cumsum.max(axis=1)

        y_cumsum_min = y_cumsum.min(axis=1)

        scores = where(
            absolute(y_cumsum_min) < absolute(y_cumsum_max), y_cumsum_max, y_cumsum_min
        )

    else:

        scores = y_cumsum.sum(axis=1)

    return scores
