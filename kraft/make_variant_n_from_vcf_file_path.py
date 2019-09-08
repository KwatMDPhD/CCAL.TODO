from pandas import read_csv, value_counts

from .flatten_nested_iterable import flatten_nested_iterable
from .make_variant_n_from_vcf_row import make_variant_n_from_vcf_row
from .VCF_COLUMNS import VCF_COLUMNS


def make_variant_n_from_vcf_file_path(vcf_file_path, use_only_pass=True):

    vcf = read_csv(vcf_file_path, sep="\t", comment="#", header=None, low_memory=False)

    filter_column = vcf.iloc[:, VCF_COLUMNS.index("FILTER")]

    if use_only_pass:

        is_pass = filter_column == "PASS"

        if use_only_pass and not is_pass.any():

            raise ValueError("There is no PASS variant.")

        vcf = vcf[is_pass]

    variant_n = value_counts(
        flatten_nested_iterable(vcf.apply(make_variant_n_from_vcf_row, axis=1))
    )

    variant_n.index.name = "Variant"

    variant_n.name = "N"

    return variant_n
