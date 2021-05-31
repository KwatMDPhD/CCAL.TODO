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
        op=op,
    )

    return leaves_list(link), fcluster(link, n_cl, criterion=cr)


def _get_coclustering_distance(
    gr_po_cl
):

    pa_ = asarray(triu_indices(gr_po_cl.shape[0], k=1)).T

    n_pa = pa_.size

    di_ = full(n_pa, 0)

    for iepa in range(n_pa):

        gr_popa_cl = gr_po_cl[pa_[iepa]]

        n_tr = 0

        n_co = 0

        for iecl in range(gr_po_cl.shape[1]):

            cl1, cl2 = gr_popa_cl[:, iecl]

            if not (isnan(cl1) or isnan(cl2)):

                n_tr += 1

                n_co += int(cl1 == cl2)

        di_[iepa] = 1 - n_co / n_tr

    return squareform(di_)


def cluster_clusterings(
    nu_po_di, n_cl, n_cl=100, random_seed=RANDOM_SEED, **kwarg_
):

    point_n = nu_po_di.shape[0]

    sample_n = int(point_n * SAMPLE_FRACTION)

    gr_po_cl = full([point_n, n_cl], nan)

    seed(seed=random_seed)

    for index in range(n_cl):

        index_ = choice(point_n, size=sample_n, replace=False)

        gr_po_cl[index_, index] = cluster(nu_po_di[index_], n_cl=n_cl, **kwarg_)[1]

    return cluster(_get_coclustering_distance(gr_po_cl), n_cl=n_cl, **kwarg_)
