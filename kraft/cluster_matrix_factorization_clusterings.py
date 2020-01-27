from numpy import full, nan
from pandas import DataFrame, Index, Series

from .cluster import cluster
from .compute_coclustering_distance import compute_coclustering_distance
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .factorize_matrix_by_update import factorize_matrix_by_update
from .plot_heat_map import plot_heat_map
from .plot_matrix_factorization import plot_matrix_factorization
from .RANDOM_SEED import RANDOM_SEED


def cluster_matrix_factorization_clusterings(
    dataframe,
    n_cluster,
    n_clustering=100,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    plot=True,
):

    clustering_x_w_point = full((n_clustering, dataframe.shape[0]), nan)

    clustering_x_h_point = full((n_clustering, dataframe.shape[1]), nan)

    n_per_print = max(1, n_clustering // 10)

    for clustering_index in range(n_clustering):

        if clustering_index % n_per_print == 0:

            print(
                "\t(r={}) {}/{}...".format(
                    n_cluster, clustering_index + 1, n_clustering
                )
            )

        w, h, e = factorize_matrix_by_update(
            dataframe.values,
            n_cluster,
            n_iteration=n_iteration,
            random_seed=random_seed + clustering_index,
        )

        if clustering_index == 0:

            w_0 = w

            h_0 = h

            e_0 = e

            factor_index = Index(
                ("r{}_f{}".format(n_cluster, i) for i in range(n_cluster)),
                name="Factor",
            )

            w_0 = DataFrame(w_0, index=dataframe.index, columns=factor_index)

            h_0 = DataFrame(h_0, index=factor_index, columns=dataframe.columns)

            if plot:

                plot_matrix_factorization((w_0,), (h_0,))

        clustering_x_w_point[clustering_index] = w.argmax(axis=1)

        clustering_x_h_point[clustering_index] = h.argmax(axis=0)

    w_point_cluster = Series(
        cluster(compute_coclustering_distance(clustering_x_w_point), n_cluster),
        name="Cluster",
        index=dataframe.index,
    )

    h_point_cluster = Series(
        cluster(compute_coclustering_distance(clustering_x_h_point), n_cluster),
        name="Cluster",
        index=dataframe.columns,
    )

    if plot:

        annotation_colorscale = DATA_TYPE_COLORSCALE["categorical"]

        plot_heat_map(
            dataframe,
            row_annotations=w_point_cluster,
            row_annotation_colorscale=annotation_colorscale,
            column_annotations=h_point_cluster,
            column_annotation_colorscale=annotation_colorscale,
            layout={"title": {"text": "Clustering ({} cluster)".format(n_cluster)}},
        )

    return (
        w_0,
        h_0,
        e_0,
        w_point_cluster,
        h_point_cluster,
    )
