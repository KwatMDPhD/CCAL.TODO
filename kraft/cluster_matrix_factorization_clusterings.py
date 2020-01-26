from numpy import full, nan
from pandas import DataFrame, Index, Series

from .cluster import cluster
from .compute_coclustering_distance import compute_coclustering_distance
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .factorize_matrix_by_nmf import factorize_matrix_by_nmf
from .plot_heat_map import plot_heat_map
from .plot_matrix_factorization import plot_matrix_factorization
from .RANDOM_SEED import RANDOM_SEED


def cluster_matrix_factorization_clusterings(
    dataframe,
    r,
    n_clustering=10,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    plot=True,
    directory_path=None,
):

    clustering_x_w_element = full((n_clustering, dataframe.shape[0]), nan)

    clustering_x_h_element = full((n_clustering, dataframe.shape[1]), nan)

    n_per_print = max(1, n_clustering // 10)

    for clustering in range(n_clustering):

        if clustering % n_per_print == 0:

            print("\t(r={}) {}/{}...".format(r, clustering + 1, n_clustering))

        w, h, e = factorize_matrix_by_nmf(
            dataframe.values,
            r,
            n_iteration=n_iteration,
            random_seed=random_seed + clustering,
        )

        if clustering == 0:

            w_0 = w

            h_0 = h

            e_0 = e

            index_factors = Index(
                ("r{}_f{}".format(r, i) for i in range(r)), name="Factor"
            )

            w_0 = DataFrame(w_0, index=dataframe.index, columns=index_factors)

            h_0 = DataFrame(h_0, index=index_factors, columns=dataframe.columns)

            w_0.to_csv("{}/w.tsv".format(directory_path), sep="\t")

            h_0.to_csv("{}/h.tsv".format(directory_path), sep="\t")

            if plot:

                plot_matrix_factorization((w_0,), (h_0,), directory_path)

        clustering_x_w_element[clustering, :] = w.argmax(axis=1)

        clustering_x_h_element[clustering, :] = h.argmax(axis=0)

    w_element_cluster = cluster(
        compute_coclustering_distance(clustering_x_w_element), r
    )

    w_element_cluster = Series(w_element_cluster, name="Cluster", index=dataframe.index)

    h_element_cluster = cluster(
        compute_coclustering_distance(clustering_x_h_element), r
    )

    h_element_cluster = Series(
        h_element_cluster, name="Cluster", index=dataframe.columns
    )

    if plot:

        annotation_colorscale = DATA_TYPE_COLORSCALE["categorical"]

        plot_heat_map(
            dataframe,
            row_annotations=w_element_cluster,
            row_annotation_colorscale=annotation_colorscale,
            column_annotations=h_element_cluster,
            column_annotation_colorscale=annotation_colorscale,
            layout={"title": {"text": "MFCC r={}".format(r)}},
            html_file_path="{}/dataframe_cluster.html".format(directory_path),
        )

    return (
        w_0,
        h_0,
        e_0,
        w_element_cluster,
        h_element_cluster,
    )
