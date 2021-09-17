from numpy import full, nan
from numpy.random import choice, seed

from ..constant import RANDOM_SEED, SAMPLE_FRACTION
from .cluster import cluster
from .get_coclustering_distance import get_coclustering_distance


def cluster_consensus(nu_po_di, n_gr, n_tr=100, ra=RANDOM_SEED, **ke_ar):

    n_po = nu_po_di.shape[0]

    gr_po_tr = full([n_po, n_tr], nan)

    n_sa = int(n_po * SAMPLE_FRACTION)

    seed(seed=ra)

    for iet in range(n_tr):

        iep_ = choice(n_po, size=n_sa, replace=False)

        gr_po_tr[iep_, iet] = cluster(nu_po_di[iep_], n_gr=n_gr, **ke_ar)[1]

    return cluster(get_coclustering_distance(gr_po_tr), n_gr=n_gr, **ke_ar)
