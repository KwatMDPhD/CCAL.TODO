from .VCF_COLUMNS import VCF_COLUMNS
from .get_vcf_info_ann import get_vcf_info_ann


def make_variant_n_from_vcf_row(vcf_row):

    info = vcf_row[VCF_COLUMNS.index("INFO")]

    gene_names = get_vcf_info_ann(info, "gene_name")

    effects = get_vcf_info_ann(info, "effect")

    return set(
        "{} ({})".format(gene_name, effect)
        for gene_name, effect in zip(gene_names, effects)
    )
