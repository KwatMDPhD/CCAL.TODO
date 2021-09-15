from scipy.cluster.hierarchy import fcluster, leaves_list, linkage


def cluster(
    nu_po_di,
    di="euclidean",
    li="ward",
    op=False,
    n_gr=0,
    cr="maxclust",
):

    li = linkage(
        nu_po_di,
        metric=di,
        method=li,
        optimal_ordering=op,
    )

    return leaves_list(li), fcluster(li, n_gr, criterion=cr)
