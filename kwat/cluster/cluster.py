from scipy.cluster.hierarchy import fcluster, leaves_list, linkage


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
