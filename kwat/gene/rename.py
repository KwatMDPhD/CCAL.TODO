from numpy import full, sort, unique

from ..dictionary import clean
from ._map_cg_to_gene import _map_cg_to_gene
from ._map_ens_to_gene import _map_ens_to_gene
from ._map_hgnc_to_gene import _map_hgnc_to_gene


def rename(na_, ke=True):

    n_na = len(na_)

    ge_ = full(n_na, "", dtype=object)

    an_ge = clean(
        {
            **_map_hgnc_to_gene(),
            **_map_ens_to_gene(),
            **_map_cg_to_gene(),
        }
    )

    n_su = 0

    fa_ = []

    for ie, na in enumerate(na_):

        if na.startswith("ENST") or na.startswith("ENSG"):

            na = na.split(sep=".", maxsplit=1)[0]

        if na in an_ge:

            ge = an_ge[na]

            n_su += 1

        else:

            if ke:

                ge = na

            else:

                ge = None

            fa_.append(na)

        ge_[ie] = ge

    fa_ = sort(unique(fa_))

    n_fa = fa_.size

    print(
        "Renamed {} ({:.2%}) failed {} ({:.2%})".format(
            n_su, n_su / n_na, n_fa, n_fa / n_na
        )
    )

    return ge_, fa_
