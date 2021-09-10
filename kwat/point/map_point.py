from numpy import apply_along_axis
from sklearn.manifold import MDS

from ..array import normalize
from ..constant import random_seed


def map_point(di_po_po, n_di, ra=random_seed, **ke_va):

    nu_po_di = MDS(
        n_components=n_di, random_state=ra, dissimilarity="precomputed", **ke_va
    ).fit_transform(di_po_po)

    for ie in range(n_di):

        nu_po_di[:, ie] = normalize(nu_po_di[:, ie], "0-1")

    # TODO
    assert (nu_po_di == apply_along_axis(normalize, 1, nu_po_di, "0-1")).all()

    return nu_po_di
