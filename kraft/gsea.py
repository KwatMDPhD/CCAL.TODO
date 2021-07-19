def _normalize_sample(input_dataset, sample_normalization):

    return nu_ge_sa


def _select_gene_sets(
    gene_sets,
    max_gene_set_size,
    min_gene_set_size,
):

    return


def _combine_up_dn(en_se_sa):

    return en_se_sa


def single_sample_gsea(
    input_dataset,
    gene_sets,
    max_gene_set_size=500,
    min_gene_set_size=5,
    enrichment_metric="ks",  # auc, js, ...
    weight_exponent=1,
    sample_normalization="z",  # None, rank, log_rank, z, log
    combine_and_add_up_dn_entries=True,
    output_file="",
):

    nu_ge_sa = _normalize_sample(input_dataset, sample_normalization)

    se_el_ = _select_gene_sets(gene_sets, max_gene_set_size, min_gene_set_size)

    en_se_sa = score_samples_and_sets(nu_ge_sa, se_el_, me=enrichment_metric)

    if combine_and_add_up_dn_entries:

        en_se_sa = _combine_up_dn(en_se_sa)

    if output_file != "":

        en_se_sa.to_csv(output_file, "\t")

    return en_se_sa


def gsea(
    input_dataset,
    phenotype_labels,
    output_folder,
    gene_sets,
    max_gene_set_size=500,
    min_gene_set_size=5,
    enrichment_metric="ks",  # auc, js, ...
    weight_exponent=1,
    metric_for_ranking_genes="s2r",  # t, tm, r, rm, d, dm, l, lm, p, c, ic,
    gene_list_sorting_mode="real",  # abs
    permutaion_type="phenotype",  # gene_set
    num_permutations=1000,
    num_of_top_gene_sets_for_plots=25,
    additional_gene_sets_for_plots=None,
    combine_and_add_up_dn_entries=True,
    sample_normalization="z",  # None, rank, log_rank, z, log
    random_seed=1729,
):

    nu_ge_sa = _normalize_sample(input_dataset, sample_normalization)

    # Rank

    se_el_ = _select_gene_sets(gene_sets, max_gene_set_size, min_gene_set_size)

    en_ = _score_sample_and_sets(sc_, se_el_, enrichment_metric)

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
    enrichment_metric="ks",  # auc, js, ...
    weight_exponent=1,
    gene_list_sorting_mode="real",  # abs
    permutaion_type="phenotype",
    num_permutations=1000,
    num_of_top_gene_sets_for_plots=25,
    additional_gene_sets_for_plots=None,
    combine_and_add_up_dn_entries=True,
    random_seed=1729,
):

    se_el_ = _select_gene_sets(gene_sets, max_gene_set_size, min_gene_set_size)

    en_ = _score_sample_and_sets(input_gene_scores, se_el_, enrichment_metric)

    # Get null scores

    # Compute p-values and q-values

    # Plot

    # Save gene_set.tsv

    return
