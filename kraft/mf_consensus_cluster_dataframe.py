from os.path import join

from numpy import full, nan
from pandas import DataFrame, Index, Series

from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .cluster_matrix import cluster_matrix
from .mf_by_multiplicative_update import mf_by_multiplicative_update
from .nmf_by_sklearn import nmf_by_sklearn
from .plot_heat_map import plot_heat_map
from .RANDOM_SEED import RANDOM_SEED


def mf_consensus_cluster_dataframe(
    dataframe,
    k,
    mf_function="nmf_by_sklearn",
    n_clustering=10,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_w=True,
    plot_h=True,
    plot_dataframe=True,
    directory_path=None,
):

    print("MFCC K={}...".format(k))

    clustering_x_w_element = full((n_clustering, dataframe.shape[0]), nan)

    clustering_x_h_element = full((n_clustering, dataframe.shape[1]), nan)

    n_per_print = max(1, n_clustering // 10)

    if mf_function == "mf_by_multiplicative_update":

        mf_function = mf_by_multiplicative_update

    elif mf_function == "nmf_by_sklearn":

        mf_function = nmf_by_sklearn

    for clustering in range(n_clustering):

        if clustering % n_per_print == 0:

            print("\t(K={}) {}/{}...".format(k, clustering + 1, n_clustering))

        w, h, e = mf_function(
            dataframe.values,
            k,
            n_iteration=n_iteration,
            random_seed=random_seed + clustering,
        )

        if clustering == 0:

            w_0 = w

            h_0 = h

            e_0 = e

            factors = Index(("Factor{}".format(i) for i in range(k)), name="Factor")

            w_0 = DataFrame(w_0, index=dataframe.index, columns=factors)

            h_0 = DataFrame(h_0, index=factors, columns=dataframe.columns)

            if directory_path is not None:

                w_0.to_csv(join(directory_path, "w.tsv"), sep="\t")

                h_0.to_csv(join(directory_path, "h.tsv"), sep="\t")

            if plot_w:

                print("Plotting w...")

                if directory_path is None:

                    html_file_path = None

                else:

                    html_file_path = join(directory_path, "w.html")

                plot_heat_map(
                    w_0.iloc[cluster_matrix(w_0.values, 0)],
                    title={"text": "MF K={} W".format(k)},
                    xaxis={"title": {"text": w_0.columns.name}},
                    yaxis={"title": {"text": w_0.index.name}},
                    html_file_path=html_file_path,
                )

            if plot_h:

                print("Plotting h...")

                if directory_path is None:

                    html_file_path = None

                else:

                    html_file_path = join(directory_path, "h.html")

                plot_heat_map(
                    h_0.iloc[:, cluster_matrix(h_0.values, 1)],
                    title={"text": "MF K={} H".format(k)},
                    xaxis={"title": {"text": h_0.columns.name}},
                    yaxis={"title": {"text": h_0.index.name}},
                    html_file_path=html_file_path,
                )

        clustering_x_w_element[clustering, :] = w.argmax(axis=1)

        clustering_x_h_element[clustering, :] = h.argmax(axis=0)

    w_element_cluster, w_element_cluster__ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_w_element, k, linkage_method
    )

    w_element_cluster = Series(w_element_cluster, name="Cluster", index=dataframe.index)

    h_element_cluster, h_element_cluster__ccc = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_h_element, k, linkage_method
    )

    h_element_cluster = Series(
        h_element_cluster, name="Cluster", index=dataframe.columns
    )

    if plot_dataframe:

        print("Plotting dataframe.clustered...")

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = join(directory_path, "dataframe.cluster.html")

        w_element_cluster_sorted = w_element_cluster.sort_values()

        h_element_cluster_sorted = h_element_cluster.sort_values()

        dataframe = dataframe.loc[
            w_element_cluster_sorted.index, h_element_cluster_sorted.index
        ]

        plot_heat_map(
            dataframe,
            row_annotations=w_element_cluster_sorted,
            column_annotations=h_element_cluster_sorted,
            title={"text": "MFCC K={}".format(k)},
            xaxis={"title": {"text": dataframe.columns.name}},
            yaxis={"title": {"text": dataframe.index.name}},
            html_file_path=html_file_path,
        )

    return (
        w_0,
        h_0,
        e_0,
        w_element_cluster,
        w_element_cluster__ccc,
        h_element_cluster,
        h_element_cluster__ccc,
    )
