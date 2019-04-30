from pandas import read_csv, value_counts

from kraft import VCF_COLUMNS, flatten_nested_iterable

from .make_variant_n_from_vcf_row import make_variant_n_from_vcf_row


def make_variant_n_from_vcf_file_path(vcf_file_path):

    vcf = read_csv(vcf_file_path, sep="\t", comment="#", header=None, low_memory=False)

    filter_column = vcf.iloc[:, VCF_COLUMNS.index("FILTER")]

    if "PASS" in filter_column.values:

        print("Using only rows with 'PASS' ...")

        vcf = vcf[filter_column == "PASS"]

    else:

        print("Using all rows because there is not a row with 'PASS' ...")

    variant_n = value_counts(
        flatten_nested_iterable(vcf.apply(make_variant_n_from_vcf_row, axis=1))
    )

    variant_n.index.name = "Variant"

    variant_n.name = "N"

    return variant_n
