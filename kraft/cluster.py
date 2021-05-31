from numpy import asarray, full, isnan, nan, triu_indices
from numpy.random import choice, seed
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .CONSTANT import RANDOM_SEED, SAMPLE_FRACTION


def cluster(
    nu_po_di,
    di="euclidean",
    li="ward",
    op=False,
    n_cl=0,
    cr="maxclust",
):

    link = linkage(
        nu_po_di,
        metric=di,
        method=li,
        optimal_ordering=op,
    )

    return leaves_list(link), fcluster(link, n_cl, criterion=cr)


def _get_coclustering_distance(cl_po_tr):

    pa_ = asarray(triu_indices(cl_po_tr.shape[0], k=1)).T

    n_pa = pa_.shape[0]

    di_ = full(n_pa, 0)

    n_cl = cl_po_tr.shape[1]

    for iepa in range(n_pa):

        cl_popa_cl = cl_po_tr[pa_[iepa]]

        n_tr = 0

        n_co = 0

        for iecl in range(n_cl):

            cl1, cl2 = cl_popa_cl[:, iecl]

            if not (isnan(cl1) or isnan(cl2)):

                n_tr += 1

                n_co += int(cl1 == cl2)

        di_[iepa] = 1 - n_co / n_tr

    return squareform(di_)


def cluster_consensus(nu_po_di, n_cl, n_tr=100, ra=RANDOM_SEED, **ke_):

    n_po = nu_po_di.shape[0]

    cl_po_tr = full([n_po, n_tr], nan)

    n_ch = int(n_po * SAMPLE_FRACTION)

    seed(ra)

    for ie in range(n_tr):

        ie_ = choice(n_po, n_ch, False)

        cl_po_tr[ie_, ie] = cluster(nu_po_di[ie_], n_cl=n_cl, **ke_)[1]

    return cluster(_get_coclustering_distance(cl_po_tr), n_cl=n_cl, **ke_)
