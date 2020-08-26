from numpy import asarray, full, isnan, nan, triu_indices
from numpy.random import choice, seed
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .CONSTANT import RANDOM_SEED, SAMPLE_FRACTION


def cluster(
    point_x_dimension,
    distance_function="euclidean",
    linkage_method="ward",
    optimal_ordering=False,
    cluster_number=0,
    criterion="maxclust",
):

    link = linkage(
        point_x_dimension,
        metric=distance_function,
        method=linkage_method,
        optimal_ordering=optimal_ordering,
    )

    leaf_index_ = leaves_list(link)

    cluster_ = fcluster(link, cluster_number, criterion=criterion) - 1

    return leaf_index_, cluster_


def _get_coclustering_distance(point_x_clustering):

    pair_ = asarray(triu_indices(point_x_clustering.shape[0], k=1)).T

    pair_number = pair_.size

    clustering_number = point_x_clustering.shape[1]

    distance_ = full(pair_number, 0)

    for pair_index in range(pair_number):

        pair_x_clustering = point_x_clustering[pair_[pair_index]]

        try_number = 0

        cocluster_number = 0

        for clustering_index in range(clustering_number):

            cluster_0, cluster_1 = pair_x_clustering[:, clustering_index]

            if not (isnan(cluster_0) or isnan(cluster_1)):

                try_number += 1

                cocluster_number += int(cluster_0 == cluster_1)

        distance_[pair_index] = 1 - cocluster_number / try_number

    return squareform(distance_)


def cluster_clusterings(
    point_x_dimension,
    cluster_number,
    clustering_number=100,
    random_seed=RANDOM_SEED,
    **kwarg_,
):

    point_number = point_x_dimension.shape[0]

    clustering_point_number = int(point_number * SAMPLE_FRACTION)

    point_x_clustering = full((point_number, clustering_number), nan)

    seed(seed=random_seed)

    for clustering_index in range(clustering_number):

        point_index_ = choice(point_number, size=clustering_point_number, replace=False)

        point_x_clustering[point_index_, clustering_index] = cluster(
            point_x_dimension[point_index_], cluster_number=cluster_number, **kwarg_
        )[1]

    return cluster(
        _get_coclustering_distance(point_x_clustering),
        cluster_number=cluster_number,
        **kwarg_,
    )
