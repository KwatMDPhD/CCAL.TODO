from os.path import join

from numpy import full, nan
from numpy.random import randint, seed
from pandas import DataFrame, Index, Series
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .make_binary_dataframe_from_categorical_series import (
    make_binary_dataframe_from_categorical_series,
)
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


def hierarchical_consensus_cluster_dataframe(
    dataframe,
    k,
    axis,
    distance__element_x_element=None,
    distance_function="euclidean",
    n_clustering=10,
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_dataframe=True,
    directory_path=None,
):

    if axis == 1:

        dataframe = dataframe.T

    if distance__element_x_element is None:

        print(
            f"Computing distance__element_x_element distance with {distance_function} ..."
        )

        distance__element_x_element = DataFrame(
            squareform(pdist(dataframe.values, distance_function)),
            index=dataframe.index,
            columns=dataframe.index,
        )

        if directory_path is not None:

            distance__element_x_element.to_csv(
                join(directory_path, "distance.element_x_element.tsv"), sep="\t"
            )

    print(f"HCC K={k} ...")

    clustering_x_element = full((n_clustering, dataframe.index.size), nan)

    n_per_print = max(1, n_clustering // 10)

    seed(seed=random_seed)

    for clustering in range(n_clustering):

        if clustering % n_per_print == 0:

            print(f"\t(K={k}) {clustering + 1}/{n_clustering} ...")

        random_elements_with_repeat = randint(
            0, high=dataframe.index.size, size=dataframe.index.size
        )

        clustering_x_element[clustering, random_elements_with_repeat] = fcluster(
            linkage(
                squareform(
                    distance__element_x_element.iloc[
                        random_elements_with_repeat, random_elements_with_repeat
                    ]
                ),
                method=linkage_method,
            ),
            k,
            criterion="maxclust",
        )

    element_cluster, element_cluster__ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_element, k, linkage_method
    )

    element_cluster = Series(element_cluster, name="Cluster", index=dataframe.index)

    cluster_x_element = make_binary_dataframe_from_categorical_series(element_cluster)

    cluster_x_element.index = Index(
        (f"Cluster{i}" for i in cluster_x_element.index),
        name=cluster_x_element.index.name,
    )

    if directory_path is not None:

        cluster_x_element.to_csv(
            join(directory_path, "cluster_x_element.tsv"), sep="\t"
        )

    if axis == 1:

        dataframe = dataframe.T

    if plot_dataframe:

        print("Plotting dataframe.clustered ...")

        element_cluster_sorted = element_cluster.sort_values()

        plot_heat_map_keyword_arguments = {}

        if axis == 0:

            dataframe = dataframe.loc[element_cluster_sorted.index]

            plot_heat_map_keyword_arguments["row_annotation"] = element_cluster_sorted

        elif axis == 1:

            dataframe = dataframe[element_cluster_sorted.index]

            plot_heat_map_keyword_arguments[
                "column_annotation"
            ] = element_cluster_sorted

        file_name = "dataframe.cluster.html"

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = join(directory_path, file_name)

        plot_heat_map(
            dataframe,
            title_text=f"HCC K={k}",
            xaxis_title_text=dataframe.columns.name,
            yaxis_title_text=dataframe.index.name,
            html_file_path=html_file_path,
            **plot_heat_map_keyword_arguments,
        )

    return element_cluster, element_cluster__ccc
