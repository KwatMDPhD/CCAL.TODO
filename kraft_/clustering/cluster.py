from scipy.cluster.hierarchy import fcluster, leaves_list, linkage


def cluster(
    point_x_dimension,
    distance_function="euclidean",
    linkage_method="ward",
    optimal_ordering=False,
    n_cluster=None,
    criterion="maxclust",
):

    z = linkage(
        point_x_dimension,
        metric=distance_function,
        method=linkage_method,
        optimal_ordering=optimal_ordering,
    )

    leave_index = leaves_list(z)

    if n_cluster is None:

        clusters = None

    else:

        clusters = fcluster(z, n_cluster, criterion=criterion) - 1

    return leave_index, clusters
