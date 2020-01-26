from numpy import arange, full, nan
from numpy.random import choice, seed
from pandas import Series

from .cluster import cluster
from .compute_coclustering_distance import compute_coclustering_distance
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
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

    seed(seed=random_seed)

    point_x_dimension = dataframe.values

    point_index = arange(n_point)

    n_choice = int(0.64 * n_point)

    for clustering_index in range(n_clustering):

        if clustering_index % n_per_print == 0:

            print("\t{}/{}...".format(clustering_index + 1, n_clustering))

        choice_index = choice(point_index, size=n_choice, replace=False)

        clustering_x_point[clustering_index, choice_index] = cluster(
            point_x_dimension[choice_index], n_cluster,
        )

    point_cluster = Series(
        cluster(compute_coclustering_distance(clustering_x_point), n_cluster),
        name="Cluster",
        index=dataframe.index,
    )

    if axis == 1:

        dataframe = dataframe.T

    if plot:

        if axis == 0:

            row_column = "row"

        elif axis == 1:

            row_column = "column"

        plot_heat_map(
            dataframe,
            layout={"title": {"text": "HCC r={}".format(n_cluster)}},
            **{
                "{}_annotations".format(row_column): point_cluster,
                "{}_annotation_colorscale".format(row_column): DATA_TYPE_COLORSCALE[
                    "categorical"
                ],
            },
        )

    return point_cluster
