from pandas import read_csv, value_counts

from ..iterable import flatten
from .COLUMNS import COLUMNS
from .make_variant_n_from_vcf_row import make_variant_n_from_vcf_row


def make_variant_n_from_vcf_file_path(pa):

    da = read_csv(pa, sep="\t", comment="#", header=None, low_memory=False)

    fi = da.iloc[:, COLUMNS.index("FILTER")]

    if "PASS" in fi.values:

        print("Using only 'PASS' rows...")

        da = da.loc[fi == "PASS", :]

    else:

        print("There is no 'PASS' rows so using all rows...")

    va_co = value_counts(flatten(da.apply(make_variant_n_from_vcf_row, axis=1)))

    va_co.index.name = "Variant"

    va_co.name = "N"

    return va_co
