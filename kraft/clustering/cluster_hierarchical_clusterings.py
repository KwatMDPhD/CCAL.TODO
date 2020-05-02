from numpy import arange, full, nan
from numpy.random import choice, seed
from pandas import Series

from .cluster import cluster
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .get_coclustering_distance import get_coclustering_distance
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


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

    n_choice = int(0.64 * n_point)

    seed(seed=random_seed)

    for clustering_index in range(n_clustering):

        if clustering_index % n_per_print == 0:

            print("{}/{}...".format(clustering_index + 1, n_clustering))

        point_index_ = choice(point_index, size=n_choice, replace=False)

        clustering_x_point[clustering_index, point_index_] = cluster(
            point_x_dimension[point_index_], n_cluster=n_cluster,
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
                "{}_annotation_colorscale".format(str_): DATA_TYPE_COLORSCALE[
                    "categorical"
                ],
            },
        )

    return Series(clusters, name="Cluster", index=dataframe.index)
