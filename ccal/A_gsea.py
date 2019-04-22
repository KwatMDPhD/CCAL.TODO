"""
Alex Wenzel's implementation of GSEA including
fast permutation scores ported from desktop
Java code
"""

from numpy import asarray, nan
from pandas import Series

from ccal.A_gsea_get_enrichment import A_gsea_get_enrichment
from ccal.A_gsea_get_pval_fdr import A_gsea_get_pval_fdr

def A_gsea(
    gene_x_sample,
    phenotypes,
    gene_sets,
    function,
    statistic="ks",
    n_permutation=None,
    permuting="gene",
    plot=True,
    title=None,
    gene_score_name=None,
    annotation_text_font_size=16,
    annotation_text_width=88,
    annotation_text_yshift=64,
    html_file_path=None,
):
    gene_score = Series(
        gene_x_sample.apply(function, axis=1, args=(asarray(phenotypes),)),
        index=gene_x_sample.index
    )
    gene_score = gene_score.sort_values(ascending=False)

    escores, mtdata = A_gsea_get_enrichment(gene_score, gene_sets)

    results = A_gsea_get_pval_fdr(
        escores,
        gene_x_sample,
        phenotypes,
        gene_sets,
        gene_score,
        function,
        n_permutation=n_permutation,
        permutation_method=permuting
    )

    return results
