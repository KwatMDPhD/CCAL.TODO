"""
Fast version of the weighted KS statistic ported
from the desktop Java code
"""

import numpy as np
import pandas as pd

def _get_gs_mapping(genes, gene_sets):
    members_map = {g: [] for g in genes}
    for gs_name, gs in gene_sets.iterrows():
        gs_short = gs.dropna(how='any')
        for gene in set(gs_short).intersection(genes):
            members_map[gene].append(gs_name)
    return members_map

def _batch_update(last_hit, last_miss, num_updates, rank_len, gs_len):
    pm_update = 1.0/(rank_len-gs_len)
    ph = [last_hit]*(num_updates)
    pm = [last_miss+pm_update]
    for _ in range(num_updates-1):
        pm.append(pm[-1]+pm_update)
    return ph, pm

def _get_escore_from_mtdata(mtdata):
    mt_max = max(mtdata)
    mt_min = min(mtdata)
    if abs(mt_max) > abs(mt_min): ##positive enrichment score
        return mt_max
    else:
        return mt_min

def A_gsea_get_enrichment(ranking, gene_sets, power=1.0):
    ## Count how many genes are being analyzed
    rank_len = ranking.shape[0]
    gs_mapping = _get_gs_mapping(ranking.index, gene_sets)

    ## Initialize dictionaries to keep track of the last time a
    ## gene set was seen and how many genes are in each set
    all_gs_names = list(gene_sets.index)
    indices_last_seen ={gs_name: -1 for gs_name in all_gs_names}
    get_gs_size = lambda x: float(gene_sets.loc[x,:].dropna(how='any').shape[0])
    gs_lens = {gs_name: get_gs_size(gs_name) for gs_name in all_gs_names}

    ## Initialize dictionaries to track the cumulative hit/miss scores
    ## for each gene set. For gene set gs, cum_ph[gs]-cum_pm[gs] gives the
    ## mountain plot for that gene set
    cum_ph = {gs_name: [0.0] for gs_name in all_gs_names}
    cum_pm = {gs_name: [0.0] for gs_name in all_gs_names}

    ## Make a pass through the raning and store the sum of absolute
    ## correlation scores for each gene set
    cum_NR = {gs_name: 0.0 for gs_name in all_gs_names}
    for indx, corr in enumerate(ranking):
        gene = ranking.index[indx]
        corr = abs(corr)
        #for gs_name in _get_member_sets(gene, gene_sets):
        for gs_name in gs_mapping[gene]:
            cum_NR[gs_name] += corr

    ## Make a pass through the ranking and update each gene set
    ## asociated with the gene in the ranking
    for indx, corr in enumerate(ranking):
        gene = ranking.index[indx]
        #for gs_name in _get_member_sets(gene, gene_sets):
        for gs_name in gs_mapping[gene]:
            ## If this gene set was last updated more than 1 gene ago
            ## perform a batch update
            last_hit = cum_ph[gs_name][-1]
            last_miss = cum_pm[gs_name][-1]
            if indices_last_seen[gs_name] != indx-1:
                indx_last_seen = indices_last_seen[gs_name]
                gs_len = gs_lens[gs_name]
                batch_ph, batch_pm = _batch_update(
                    last_hit, last_miss, int(indx-indx_last_seen),
                    rank_len, gs_len
                )
                cum_ph[gs_name] += batch_ph
                cum_pm[gs_name] += batch_pm

            ## Perform a single hit update
            indices_last_seen[gs_name] = indx
            last_hit = cum_ph[gs_name][-1]
            last_miss = cum_pm[gs_name][-1]
            try:
                cum_ph[gs_name].append(last_hit+(abs(corr**power)/cum_NR[gs_name]))
            except FloatingPointError:
                cum_ph[gs_name].append(last_hit)
            cum_pm[gs_name].append(last_miss)

    ## This loop iterates through all gene sets and makes sure they are up to
    ## date before returning data
    for gs_name in all_gs_names:
        gs_len = gs_lens[gs_name]
        last_hit = cum_ph[gs_name][-1]
        last_miss = cum_pm[gs_name][-1]
        indx_last_seen = indices_last_seen[gs_name]
        if rank_len-indx_last_seen-1 == 0:
            continue
        batch_ph, batch_pm = _batch_update(
            last_hit, last_miss, int(rank_len-indx_last_seen)-1,
            rank_len, gs_len
        )
        cum_ph[gs_name] += batch_ph
        cum_pm[gs_name] += batch_pm

    mtdata = {
        gs_name: list(np.array(cum_ph[gs_name])-np.array(cum_pm[gs_name]))
        for gs_name in all_gs_names
    }
    escores = pd.Series({
        gs_name: _get_escore_from_mtdata(mtdata[gs_name])
        for gs_name in all_gs_names
    })
    return escores, mtdata
