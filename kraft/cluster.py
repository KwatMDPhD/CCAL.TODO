from scipy.cluster.hierarchy import fclusterdata


def cluster(
    element_x_dimension,
    r,
    criterion="maxclust",
    metric="correlation",
    method="centroid",
):

    return (
        fclusterdata(
            element_x_dimension, r, criterion=criterion, metric=metric, method=method,
        )
        - 1
    )
