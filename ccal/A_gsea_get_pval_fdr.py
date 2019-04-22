"""
Implements permutation testing and FDR for GSEA
"""

import numpy as np
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests

from ccal.A_gsea_get_enrichment import A_gsea_get_enrichment

"""
Utilities
"""

def get_gs_len(gene_set):
    """
    Gets the non-NA length of a gene set from a CCAL gmt/dataframe
    """
    return gene_set.dropna(how='any').shape[0]

"""
Permutation functions
"""

def _permute(
    gene_x_sample,
    phenotypes,
    permutation_method
):
    """
    Returns a new gene_x_sample and phenotypes objects based on
    permutation_method. A separate function is used for gene sets permutation,
    which only has to be done once as opposed to every iteration
    """
    null_gene_x_sample = gene_x_sample.copy()
    null_phenotypes = phenotypes.copy()

    if permutation_method == "gene":
        null_index = list(null_gene_x_sample.index)
        np.random.shuffle(null_index)
        null_gene_x_sample = null_gene_x_sample.loc[null_index,:]

    elif permutation_method == "phenotype":
        np.random.shuffle(null_phenotypes)

    else:
        raise ValueError("Invalid permutation_method: "+permutation_method)

    return null_gene_x_sample, null_phenotypes

def _get_random_gene_set(
    gene_universe,
    size,
    max_size
):
    """
    Given a gene_universe, builds a random gene set with size genes
    and pads it with (max_size-size) NaNs
    """
    gs = list(np.random.choice(gene_universe, size, replace=False))
    gs = gs + [np.nan]*(max_size-size)
    return gs

def _sample_gene_sets(
    gene_sets,
    gene_universe,
    n_permutation
):
    """
    Builds a new dataframe of randomly-sampled gene sets with
    (n_permutation * gene_sets.shape[0]) null gene sets
    """
    ## Find out how many different sizes of gene sets there are
    gs_lens = {
        gs_name: get_gs_len(gene_sets.loc[gs_name,:])
        for gs_name in gene_sets.index
    }
    unique_lens = set(gs_lens.values())
    max_unique_len = max(unique_lens)

    ## For each size of gene set, randomly sample n_permutation new
    ## gene sets
    null_gene_sets_lens = {}  ##dict of {int: DataFrame}
    for gslen in unique_lens:
        null_gene_sets = pd.DataFrame({
            "null_"+str(i): _get_random_gene_set(gene_universe, gslen, max_unique_len)
            for i in range(n_permutation)
        }).T
        null_gene_sets_lens[gslen] = null_gene_sets
    return null_gene_sets_lens

def A_gsea_get_pval_fdr(
    escores,
    gene_x_sample,
    phenotypes,
    gene_sets,
    ranking,
    function,
    n_permutation=1000,
    permutation_method="phenotype"
):
    ## Get permutation results
    null_escores = {gs_name: [] for gs_name in gene_sets.index}

    if permutation_method == "gene_set":
        null_gene_sets = _sample_gene_sets(gene_sets, gene_x_sample.index, n_permutation)
        for gs_name in gene_sets.index:
            gs_len = get_gs_len(gene_sets.loc[gs_name,:])
            es, mtdata = A_gsea_get_enrichment(
                ranking, null_gene_sets[gs_len]
            )
            null_escores[gs_name] = es.values

    elif permutation_method in ("gene", "phenotype"):
        for i in range(n_permutation):
            null_gene_x_sample, null_phenotypes = _permute(
                gene_x_sample, phenotypes, permutation_method
            )
            null_ranking = pd.Series(
                gene_x_sample.apply(
                    function,
                    axis=1,
                    args=(np.asarray(null_phenotypes),)),
                    index=null_gene_x_sample.index
            )
            es, mtdata = A_gsea_get_enrichment(
                null_ranking, gene_sets
            )
            for gs_name in null_escores.keys():
                null_escores[gs_name].append(es[gs_name])
    null_escores = pd.DataFrame(null_escores).T

    ## Get normalized enrichment scores
    pos_means = {}
    neg_means = {}

    for gs_name in null_escores.index:
        vals = null_escores.loc[gs_name,:]
        pos_means[gs_name] = np.mean(vals[vals>0])
        neg_means[gs_name] = np.mean(vals[vals<0])

    norm_null_escores = {}

    for gs_name in null_escores.index:
        nes = []
        for es in null_escores.loc[gs_name,:]:
            if es > 0:
                nes.append(es/pos_means[gs_name])
            else:
                nes.append(es/abs(neg_means[gs_name]))
        norm_null_escores[gs_name] = nes

    norm_null_escores = pd.DataFrame(
        norm_null_escores
    ).T

    norm_escores = [
        x/pos_means[gs] if x > 0 else x/abs(neg_means[gs])
        for gs, x in escores.iteritems()
    ]
    norm_escores = pd.Series(norm_escores, index=escores.index)

    ## get p-values
    pvals = {gs_name: None for gs_name in null_escores.index}

    for gs_name, nes in norm_escores.iteritems():
        perms = norm_null_escores.loc[gs_name,:]
        if nes > 0:
            n_more_extr = perms[perms>nes].shape[0]
        else:
            n_more_extr = perms[perms<nes].shape[0]
        pvals[gs_name] = float(n_more_extr)/n_permutation

    pvals = pd.Series(pvals)

    adj_pvals = multipletests(pvals, method='fdr_bh')[1]
    adj_pvals = pd.Series(adj_pvals, index=pvals.index)

    results = pd.DataFrame({
        "es": escores.values,
        "nes": norm_escores.values,
        "pval": pvals.values,
        "adj_pval": adj_pvals.values
    }, index=escores.index)

    return results
