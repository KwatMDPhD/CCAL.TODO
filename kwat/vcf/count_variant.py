from numpy import apply_along_axis
from pandas import value_counts

from ..iterable import flatten
from ..string import split_and_get
from .ANN import ANN
from .COLUMN import COLUMN
from .read import read


def _get_info(io, ket):

    for sp in io.split(sep=";"):

        if "=" in sp:

            ke, va = sp.split(sep="=")

            if ke == ket:

                return va


def _get_info_ann(io, ket, n_an=None):

    an = _get_info(io, "ANN")

    if an is not None:

        ie = ANN.index(ket)

        return [split_and_get(sp, "|", ie) for sp in an.split(sep=",")[:n_an]]


def _list_variant(st_):

    io = st_[COLUMN.index("INFO")]

    return set(
        "{} ({})".format(ge, ef)
        for ge, ef in zip(_get_info_ann(io, "gene_name"), _get_info_ann(io, "effect"))
    )


def count_variant(pa):

    da = read(pa)

    print(da.shape)

    fi_ = da.iloc[:, COLUMN.index("FILTER")].values

    if "PASS" in fi_:

        print("Using only 'PASS'")

        da = da.loc[fi_ == "PASS", :]

        print(da.shape)

    else:

        print("There is no 'PASS' and using all")

    va_co = value_counts(flatten(apply_along_axis(_list_variant, 1, da.values)))

    va_co.index.name = "Variant"

    va_co.name = "N"

    return va_co
