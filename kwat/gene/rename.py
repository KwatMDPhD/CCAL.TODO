from numpy import array

from ._map_cg_to_gene import _map_cg_to_gene
from ._map_ens_to_gene import _map_ens_to_gene
from ._map_ensmus_to_gene import _map_ensmus_to_gene
from ._map_hgnc_to_gene import _map_hgnc_to_gene


def rename(na_):

    an_ge = {
        **_map_hgnc_to_gene(),
        **_map_ens_to_gene(),
        **_map_cg_to_gene(),
        **_map_ensmus_to_gene(),
    }

    ge_ = [an_ge.get(na) for na in na_]

    re_ = array([ge is not None for ge in ge_])

    n_re = re_.sum()

    n_na = len(na_)

    print("Renamed {}/{} ({:.2%})".format(n_re, n_na, n_re / n_na))

    if n_re == 0:

        return na_

    else:

        return ge_
