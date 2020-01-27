from scipy.cluster.hierarchy import fclusterdata


def cluster(
    point_x_dimension,
    n_cluster,
    distance_function="euclidean",
    linkage_method="ward",
    criterion="maxclust",
):

    return (
        fclusterdata(
            point_x_dimension,
            n_cluster,
            metric=distance_function,
            method=linkage_method,
            criterion=criterion,
        )
        - 1
    )
