from numpy import full, nan
from pandas import DataFrame, Index, Series

from .cluster_clustering_x_element_and_compute_ccc import (
    cluster_clustering_x_element_and_compute_ccc,
)
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .mf_with_multiplicative_update import mf_with_multiplicative_update
from .nmf_with_sklearn import nmf_with_sklearn
from .plot_heat_map import plot_heat_map
from .plot_mf import plot_mf
from .RANDOM_SEED import RANDOM_SEED


def mf_consensus_cluster_dataframe(
    dataframe,
    r,
    directory_path,
    mf_function="nmf_with_sklearn",
    n_clustering=10,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_heat_map_=True,
):

    clustering_x_w_element = full((n_clustering, dataframe.shape[0]), nan)

    clustering_x_h_element = full((n_clustering, dataframe.shape[1]), nan)

    n_per_print = max(1, n_clustering // 10)

    mf_function = {
        "mf_with_multiplicative_update": mf_with_multiplicative_update,
        "nmf_with_sklearn": nmf_with_sklearn,
    }[mf_function]

    for clustering in range(n_clustering):

        if clustering % n_per_print == 0:

            print("\t(r={}) {}/{}...".format(r, clustering + 1, n_clustering))

        w, h, e = mf_function(
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

            if plot_heat_map_:

                plot_mf((w_0,), (h_0,), directory_path)

        clustering_x_w_element[clustering, :] = w.argmax(axis=1)

        clustering_x_h_element[clustering, :] = h.argmax(axis=0)

    (
        w_element_cluster,
        w_element_cluster_ccc,
    ) = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_w_element, r, linkage_method
    )

    w_element_cluster = Series(w_element_cluster, name="Cluster", index=dataframe.index)

    (
        h_element_cluster,
        h_element_cluster_ccc,
    ) = cluster_clustering_x_element_and_compute_ccc(
        clustering_x_h_element, r, linkage_method
    )

    h_element_cluster = Series(
        h_element_cluster, name="Cluster", index=dataframe.columns
    )

    if plot_heat_map_:

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
        w_element_cluster_ccc,
        h_element_cluster,
        h_element_cluster_ccc,
    )
