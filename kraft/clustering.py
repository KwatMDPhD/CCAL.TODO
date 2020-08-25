from numpy import asarray, full, isnan, nan, triu_indices
from numpy.random import choice, seed
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .CONSTANT import RANDOM_SEED


def cluster(
    point_x_dimension,
    distance_function="euclidean",
    linkage_method="ward",
    optimal_ordering=False,
    n_cluster=0,
    criterion="maxclust",
):

    link = linkage(
        point_x_dimension,
        metric=distance_function,
        method=linkage_method,
        optimal_ordering=optimal_ordering,
    )

    index_ = leaves_list(link)

    cluster_ = fcluster(link, n_cluster, criterion=criterion) - 1

    return index_, cluster_


def _get_coclustering_distance(point_x_clustering):

    pair_ = asarray(triu_indices(point_x_clustering.shape[0], k=1)).T

    n_pair = pair_.size

    n_clustering = point_x_clustering.shape[1]

    distance_ = full(n_pair, 0)

    for index in range(n_pair):

        pair_x_clustering = point_x_clustering[pair_[index]]

        n_try = 0

        n_cocluster = 0

        for clustering_index in range(n_clustering):

            cluster_0, cluster_1 = pair_x_clustering[:, clustering_index]

            if not isnan(cluster_0) and not isnan(cluster_1):

                n_try += 1

                if cluster_0 == cluster_1:

                    n_cocluster += 1

        distance_[index] = 1 - n_cocluster / n_try

    return squareform(distance_)


def cluster_clusterings(
    point_x_dimension,
    n_cluster,
    n_clustering=100,
    random_seed=RANDOM_SEED,
    **keyword_arguments,
):

    n_point = point_x_dimension.shape[0]

    n_point_to_cluster = int(n_point * 0.632)

    point_x_clustering = full((n_point, n_clustering), nan)

    seed(seed=random_seed)

    for clustering_index in range(n_clustering):

        point_index_ = choice(n_point, size=n_point_to_cluster, replace=False)

        point_x_clustering[point_index_, clustering_index] = cluster(
            point_x_dimension[point_index_], n_cluster=n_cluster, **keyword_arguments
        )[1]

    return cluster(
        _get_coclustering_distance(point_x_clustering),
        n_cluster=n_cluster,
        **keyword_arguments,
    )
