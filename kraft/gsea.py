def single_sample_gsea(
    input_dataset,
    output_file,
    gene_sets,
    max_gene_set_size=500,
    min_gene_set_size=5,
    enrichment_metric="ks", # auc, js, ...
    weight_exponent=1,
    sample_normalization="z", # None, rank, log_rank, z, log
    combine_and_add_up_dn_entries=True,
):

    # Normalize each sample

    # Select gene sets

    # For each sample, score set

    # Combine up and down scores

    # Save gene_set_x_sample.tsv

    return

def gsea(
    input_dataset,
    phenotype_labels,
    output_folder,
    gene_sets,
    max_gene_set_size=500,
    min_gene_set_size=5,
    enrichment_metric="ks", # auc, js, ...
    weight_exponent=1,
    metric_for_ranking_genes="s2r", # t, tm, r, rm, d, dm, l, lm, p, c, ic,
    gene_list_sorting_mode="real", # abs
    permutaion_type="phenotype", # gene_set
    num_permutations=1000,
    num_of_top_gene_sets_for_plots=25,
    additional_gene_sets_for_plots=None,
    combine_and_add_up_dn_entries=True,
    sample_normalization="z", # None, rank, log_rank, z, log
    random_seed=1729,
):

    # Normalize each sample

    # Rank

    # Select gene sets

    # Score set

    # Get null scores

    # Compute p-values and q-values

    # Plot

    # Save gene_set.tsv

    return

def prerank_gsea(
    input_gene_scores,
    output_folder,
    gene_sets,
    max_gene_set_size=500,
    min_gene_set_size=5,
    enrichment_metric="ks", # auc, js, ...
    weight_exponent=1,
    gene_list_sorting_mode="real", # abs
    permutaion_type="phenotype",
    num_permutations=1000,
    num_of_top_gene_sets_for_plots=25,
    additional_gene_sets_for_plots=None,
    combine_and_add_up_dn_entries=True,
    random_seed=1729,
):

    # Select gene sets

    # Score set

    # Get null scores

    # Compute p-values and q-values

    # Plot

    # Save gene_set.tsv

    return
