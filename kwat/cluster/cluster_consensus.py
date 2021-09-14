from numpy import array, full, isnan, nan, triu_indices
from numpy.random import choice, seed
from scipy.spatial.distance import squareform

from ..constant import RANDOM_SEED, SAMPLE_FRACTION
from .cluster import cluster


def _get_coclustering_distance(cl_po_tr):

    pa_ = array(triu_indices(cl_po_tr.shape[0], k=1)).T

    n_pa = pa_.shape[0]

    di_ = full(n_pa, 0)

    n_to = cl_po_tr.shape[1]

    for iep in range(n_pa):

        cl_pop_tr = cl_po_tr[pa_[iep]]

        n_tr = 0

        n_co = 0

        for iet in range(n_to):

            cl1, cl2 = cl_pop_tr[:, iet]

            if not (isnan(cl1) or isnan(cl2)):

                n_tr += 1

                n_co += int(cl1 == cl2)

        di_[iep] = 1 - n_co / n_tr

    return squareform(di_)


def cluster_consensus(nu_po_di, n_cl, n_tr=100, ra=RANDOM_SEED, **ke):

    n_po = nu_po_di.shape[0]

    cl_po_tr = full([n_po, n_tr], nan)

    n_sa = int(n_po * SAMPLE_FRACTION)

    seed(seed=ra)

    for ie in range(n_tr):

        ie_ = choice(n_po, size=n_sa, replace=False)

        cl_po_tr[ie_, ie] = cluster(nu_po_di[ie_], n_cl=n_cl, **ke)[1]

    return cluster(_get_coclustering_distance(cl_po_tr), n_cl=n_cl, **ke)
