from numpy import full, nan
from numpy.random import randint, seed
from pandas import DataFrame, Series
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .make_binary_dataframe_from_categorical_series import (
    make_binary_dataframe_from_categorical_series,
)
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


def hierarchical_consensus_cluster_dataframe(
    dataframe,
    r,
    axis,
    directory_path,
    element_x_element_distance=None,
    distance_function="correlation",
    n_clustering=10,
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_dataframe=True,
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

        element_x_element_distance.to_csv(
            "{}/element_x_element_distance.tsv".format(directory_path), sep="\t"
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

        clustering_x_element[clustering, random_elements_with_repeat] = fcluster(
            linkage(
                squareform(
                    element_x_element_distance.iloc[
                        random_elements_with_repeat, random_elements_with_repeat
                    ]
                ),
                method=linkage_method,
            ),
            r,
            criterion="maxclust",
        )

    element_cluster, element_cluster_ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_element, r, linkage_method
    )

    element_cluster = Series(element_cluster, name="Cluster", index=dataframe.index)

    cluster_x_element = make_binary_dataframe_from_categorical_series(element_cluster)

    cluster_x_element.to_csv(
        "{}/cluster_x_element.tsv".format(directory_path), sep="\t"
    )

    if axis == 1:

        dataframe = dataframe.T

    if plot_dataframe:

        if axis == 0:

            row_or_column = "row"

        elif axis == 1:

            row_or_column = "column"

        plot_heat_map_keyword_arguments = {
            "{}_annotations".format(row_or_column): element_cluster,
            "{}_annotation_colorscale".format(row_or_column): DATA_TYPE_COLORSCALE[
                "categorical"
            ],
        }

        plot_heat_map(
            dataframe,
            layout={"title": {"text": "HCC r={}".format(r)}},
            html_file_path="{}/dataframe_cluster.html".format(directory_path),
            **plot_heat_map_keyword_arguments,
        )

    return element_cluster, element_cluster_ccc
