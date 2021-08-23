from numpy import array

from ._map_cg import _map_cg
from ._map_ens import _map_ens
from ._map_hgnc import _map_hgnc


def rename(na_):

    an_ge = {**_map_hgnc(), **_map_ens(), **_map_cg()}

    ge_ = [an_ge.get(na) for na in na_]

    bo_ = array([ge is not None for ge in ge_])

    n_to = bo_.size

    n_ge = bo_.sum()

    print("Named {}/{} ({:.2%})".format(n_ge, n_to, n_ge / n_to))

    if n_ge == 0:

        return na_

    else:

        return ge_
