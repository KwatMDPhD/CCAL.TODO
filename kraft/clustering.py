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
    cluster_n=0,
    criterion="maxclust",
):

    link = linkage(
        point_x_dimension,
        metric=distance_function,
        method=linkage_method,
        optimal_ordering=optimal_ordering,
    )

    return leaves_list(link), fcluster(link, cluster_n, criterion=criterion) - 1


def _get_coclustering_distance(point_x_clustering):

    pair_ = asarray(triu_indices(point_x_clustering.shape[0], k=1)).T

    pair_n = pair_.size

    clustering_n = point_x_clustering.shape[1]

    distance_ = full(pair_n, 0)

    for pair_index in range(pair_n):

        pair_x_clustering = point_x_clustering[pair_[pair_index]]

        clustering_n = 0

        cocluster_n = 0

        for clustering_index in range(clustering_n):

            cluster_0, cluster_1 = pair_x_clustering[:, clustering_index]

            if not (isnan(cluster_0) or isnan(cluster_1)):

                clustering_n += 1

                cocluster_n += int(cluster_0 == cluster_1)

        distance_[pair_index] = 1 - cocluster_n / clustering_n

    return squareform(distance_)


def cluster_clusterings(
    point_x_dimension,
    cluster_n,
    clustering_n=100,
    random_seed=RANDOM_SEED,
    **kwarg_,
):

    point_n = point_x_dimension.shape[0]

    sample_n = int(point_n * SAMPLE_FRACTION)

    point_x_clustering = full((point_n, clustering_n), nan)

    seed(seed=random_seed)

    for index in range(clustering_n):

        index_ = choice(point_n, size=sample_n, replace=False)

        point_x_clustering[index_, index] = cluster(
            point_x_dimension[index_], cluster_n=cluster_n, **kwarg_
        )[1]

    return cluster(
        _get_coclustering_distance(point_x_clustering), cluster_n=cluster_n, **kwarg_
    )
