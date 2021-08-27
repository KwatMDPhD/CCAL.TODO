from .COLUMNS import COLUMNS
from .get_vcf_info_ann import get_vcf_info_ann


def make_variant_n_from_vcf_row(ro):

    io = ro[COLUMNS.index("INFO")]

    return set(
        "{} ({})".format(ge, ef)
        for ge, ef in zip(
            get_vcf_info_ann(io, "gene_name"), get_vcf_info_ann(io, "effect")
        )
    )
