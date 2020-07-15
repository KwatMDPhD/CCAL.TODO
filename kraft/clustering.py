from numpy import apply_along_axis, arange, asarray, full, isnan, nan, triu_indices
from numpy.random import choice, seed
from pandas import Series
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from .CONSTANT import RANDOM_SEED
from .plot import DATA_TYPE_TO_COLORSCALE, plot_heat_map


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


def get_coclustering_distance(clustering_x_point):

    point_pair_indexs = tuple(zip(*triu_indices(clustering_x_point.shape[1], k=1)))

    def is_coclustered(clusters):

        return tuple(
            clusters[point_0_index] == clusters[point_1_index]
            for point_0_index, point_1_index in point_pair_indexs
        )

    n_coclustered = apply_along_axis(is_coclustered, 1, clustering_x_point).sum(axis=0)

    n = asarray(
        tuple(
            (~isnan(clustering_x_point[:, point_pair_index])).all(axis=1).sum()
            for point_pair_index in point_pair_indexs
        )
    )

    assert 0 < n.min()

    return squareform(1 - n_coclustered / n)


def cluster_hierarchical_clusterings(
    dataframe, axis, n_cluster, n_clustering=100, random_seed=RANDOM_SEED, plot=True,
):

    if axis == 1:

        dataframe = dataframe.T

    n_point = dataframe.shape[0]

    clustering_x_point = full((n_clustering, n_point), nan)

    n_per_print = max(1, n_clustering // 10)

    point_x_dimension = dataframe.values

    point_index = arange(n_point)

    n_choice = int(n_point * 0.632)

    seed(seed=random_seed)

    for clustering_index in range(n_clustering):

        if clustering_index % n_per_print == 0:

            print("{}/{}...".format(clustering_index + 1, n_clustering))

        point_index_choice = choice(point_index, size=n_choice, replace=False)

        clustering_x_point[clustering_index, point_index_choice] = cluster(
            point_x_dimension[point_index_choice], n_cluster=n_cluster,
        )[1]

    leave_index, clusters = cluster(
        get_coclustering_distance(clustering_x_point), n_cluster=n_cluster
    )

    if plot:

        dataframe = dataframe.iloc[leave_index]

        if axis == 1:

            dataframe = dataframe.T

        if axis == 0:

            str_ = "row"

        elif axis == 1:

            str_ = "column"

        plot_heat_map(
            dataframe,
            ordered_annotation=True,
            layout={"title": {"text": "Clustering ({} cluster)".format(n_cluster)}},
            **{
                "{}_annotations".format(str_): clusters[leave_index],
                "{}_annotation_colorscale".format(str_): DATA_TYPE_TO_COLORSCALE[
                    "categorical"
                ],
            },
        )

    return Series(clusters, name="Cluster", index=dataframe.index)
