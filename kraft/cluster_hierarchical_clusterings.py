from numpy import full, nan
from numpy.random import randint, seed
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform

from .cluster import cluster
from .count_coclustering import count_coclustering
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


def cluster_hierarchical_clusterings(
    dataframe,
    axis,
    r,
    element_x_element_distance=None,
    distance_function="correlation",
    n_clustering=10,
    random_seed=RANDOM_SEED,
    plot=True,
):

    if axis == 1:

        dataframe = dataframe.T

    if element_x_element_distance is None:

        print("Computing element-element distance with {}...".format(distance_function))

        element_x_element_distance = DataFrame(
            squareform(pdist(dataframe.values, distance_function)),
            index=dataframe.index,
            columns=dataframe.index,
        )

    clustering_x_element = full((n_clustering, dataframe.shape[0]), nan)

    n_per_print = max(1, n_clustering // 10)

    seed(seed=random_seed)

    for clustering in range(n_clustering):

        if clustering % n_per_print == 0:

            print("\t(r={}) {}/{}...".format(r, clustering + 1, n_clustering))

        random_elements_with_repeat = randint(
            0, high=dataframe.index.size, size=dataframe.index.size
        )

        clustering_x_element[clustering, random_elements_with_repeat] = cluster(
            element_x_element_distance.iloc[
                random_elements_with_repeat, random_elements_with_repeat
            ],
            r,
        )

    element_cluster = Series(
        cluster(count_coclustering(clustering_x_element), r),
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
            layout={"title": {"text": "HCC r={}".format(r)}},
            **{
                "{}_annotations".format(row_column): element_cluster,
                "{}_annotation_colorscale".format(row_column): DATA_TYPE_COLORSCALE[
                    "categorical"
                ],
            },
        )

    return element_cluster
