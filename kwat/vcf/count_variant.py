from numpy import apply_along_axis
from pandas import value_counts

from ..iterable import flatten
from .COLUMN import COLUMN


def _get_info(io, ke):

    for ios in io.split(sep=";"):

        if "=" in ios:

            ke2, va = ios.split(sep="=")

            if ke2 == ke:

                return va


from .ANN_KEY import ANN_KEY


def _get_info_ann(io, ke, n_an=None):

    an = _get_info(io, "ANN")

    if an is not None:

        ie = ANN_KEY.index(ke)

        return [ans.split(sep="|")[ie] for ans in an.split(sep=",")[:n_an]]


def list_variant(se):

    io = se[COLUMN.index("INFO")]

    return set(
        "{} ({})".format(ge, ef)
        for ge, ef in zip(_get_info_ann(io, "gene_name"), _get_info_ann(io, "effect"))
    )


from .read import read


def count_variant(pa):

    da = read(pa)

    fi = da.iloc[:, COLUMN.index("FILTER")].values

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
