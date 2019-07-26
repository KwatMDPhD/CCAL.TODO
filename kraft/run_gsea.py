from random import shuffle

from numpy import asarray, full, nan
from numpy.random import choice
from pandas import Series

from .compute_empirical_p_value import compute_empirical_p_value
from .run_single_sample_gsea import run_single_sample_gsea


def run_gsea(
    gene_x_sample,
    phenotypes,
    genes,
    function,
    statistic="auc",
    n_permutation=10,
    permuting="gene",
    plot=True,
    title="GSEA Mountain Plot",
    gene_score_name="Gene Score",
    annotation_text_font_size=10,
    annotation_text_width=100,
    annotation_text_yshift=50,
    html_file_path=None,
):

    print("Computing gene scores ...")

    gene_score_no_na_sorted = (
        Series(
            gene_x_sample.apply(function, axis=1, args=(asarray(phenotypes),)),
            index=gene_x_sample.index,
        )
        .dropna()
        .sort_values(ascending=False)
    )

    print(gene_score_no_na_sorted)

    print(f"Computing gene set enrichment for {genes.name} ...")

    gsea_score = run_single_sample_gsea(
        gene_score_no_na_sorted,
        genes,
        statistic=statistic,
        plot=plot,
        title_text=title,
        gene_score_name=gene_score_name,
        annotation_text_font_size=annotation_text_font_size,
        annotation_text_width=annotation_text_width,
        annotation_text_yshift=annotation_text_yshift,
        html_file_path=html_file_path,
    )

    if n_permutation == 0:

        p_value = nan

    else:

        permutation_scores = full(n_permutation, nan)

        permuting__gene_x_sample = gene_x_sample.copy()

        permuting__phenotypes = asarray(phenotypes)

        for i in range(n_permutation):

            if permuting == "phenotype":

                shuffle(permuting__phenotypes)

            elif permuting == "gene":

                permuting__gene_x_sample.index = choice(
                    permuting__gene_x_sample.index,
                    size=permuting__gene_x_sample.shape[0],
                    replace=False,
                )

            permutation_scores[i] = run_single_sample_gsea(
                Series(
                    permuting__gene_x_sample.apply(
                        function, axis=1, args=(permuting__phenotypes,)
                    ),
                    index=permuting__gene_x_sample.index,
                ),
                genes,
                statistic=statistic,
                plot=False,
            )

        p_value = min(
            compute_empirical_p_value(gsea_score, permutation_scores, "<"),
            compute_empirical_p_value(gsea_score, permutation_scores, ">"),
        )

    return gsea_score, p_value
