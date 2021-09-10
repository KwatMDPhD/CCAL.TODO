from numpy import full, nan
from numpy.random import choice, seed

from ..constant import random_seed, sample_fraction
from ._get_coclustering_distance import _get_coclustering_distance
from .cluster import cluster


def cluster_consensus(nu_po_di, n_cl, n_tr=100, ra=random_seed, **ke):

    n_po = nu_po_di.shape[0]

    cl_po_tr = full([n_po, n_tr], nan)

    n_sa = int(n_po * sample_fraction)

    seed(seed=ra)

    for ie in range(n_tr):

        ie_ = choice(n_po, size=n_sa, replace=False)

        cl_po_tr[ie_, ie] = cluster(nu_po_di[ie_], n_cl=n_cl, **ke)[1]

    return cluster(_get_coclustering_distance(cl_po_tr), n_cl=n_cl, **ke)
