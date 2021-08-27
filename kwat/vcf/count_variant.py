from numpy import apply_along_axis
from pandas import value_counts

from ..iterable import flatten
from .COLUMNS import COLUMNS
from .list_variant import list_variant
from .read import read


def count_variant(pa):

    da = read(pa)

    fi = da.iloc[:, COLUMNS.index("FILTER")].values

    print(da.shape)

    if "PASS" in fi:

        print("Using only 'PASS'...")

        da = da.loc[fi == "PASS", :]

        print(da.shape)

    else:

        print("There is no 'PASS' and using all...")

    va_co = value_counts(flatten(apply_along_axis(list_variant, 1, da.values)))

    va_co.index.name = "Variant"

    va_co.name = "N"

    return va_co
