from numpy import (
    asarray,
    full,
    isnan,
    nan,
    triu_indices,
)
from numpy.random import (
    choice,
    seed,
)
from scipy.cluster.hierarchy import (
    fcluster,
    leaves_list,
    linkage,
)
from scipy.spatial.distance import (
    squareform,
)

from .CONSTANT import (
    RANDOM_SEED,
    SAMPLE_FRACTION,
)


def cluster(
    pxd,
    distance_function="euclidean",
    linkage_method="ward",
    optimal_ordering=False,
    n_cluster=0,
    criterion="maxclust",
):

    link = linkage(
        pxd,
        metric=distance_function,
        method=linkage_method,
        optimal_ordering=optimal_ordering,
    )

    return (
        leaves_list(link),
        fcluster(
            link,
            n_cluster,
            criterion=criterion,
        )
        - 1,
    )


def _get_coclustering_distance(
    pxc,
):

    pair_ = asarray(
        triu_indices(
            pxc.shape[0],
            k=1,
        )
    ).T

    n_pair = pair_.size

    clustering_n = pxc.shape[1]

    distance_ = full(
        n_pair,
        0,
    )

    for pair_index in range(n_pair):

        pair_x_clustering = pxc[pair_[pair_index]]

        clustering_n = 0

        con_cluster = 0

        for clustering_index in range(clustering_n):

            (cluster_0, cluster_1,) = pair_x_clustering[
                :,
                clustering_index,
            ]

            if not (isnan(cluster_0) or isnan(cluster_1)):

                clustering_n += 1

                con_cluster += int(cluster_0 == cluster_1)

        distance_[pair_index] = 1 - con_cluster / clustering_n

    return squareform(distance_)


def cluster_clusterings(
    pxd,
    n_cluster,
    clustering_n=100,
    random_seed=RANDOM_SEED,
    **kwarg_,
):

    point_n = pxd.shape[0]

    sample_n = int(point_n * SAMPLE_FRACTION)

    pxc = full(
        (
            point_n,
            clustering_n,
        ),
        nan,
    )

    seed(seed=random_seed)

    for index in range(clustering_n):

        index_ = choice(
            point_n,
            size=sample_n,
            replace=False,
        )

        pxc[index_, index,] = cluster(
            pxd[index_],
            n_cluster=n_cluster,
            **kwarg_,
        )[1]

    return cluster(
        _get_coclustering_distance(pxc),
        n_cluster=n_cluster,
        **kwarg_,
    )
