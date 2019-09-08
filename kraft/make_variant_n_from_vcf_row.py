from .get_vcf_info_ann import get_vcf_info_ann
from .VCF_COLUMNS import VCF_COLUMNS


def make_variant_n_from_vcf_row(vcf_row):

    info = vcf_row[VCF_COLUMNS.index("INFO")]

    return set(
        "{} ({})".format(gene_name, effect)
        for gene_name, effect in zip(
            get_vcf_info_ann(info, "gene_name"), get_vcf_info_ann(info, "effect")
        )
    )
