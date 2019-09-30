from os.path import join

from numpy import asarray, sort
from pandas import DataFrame, Index

from .call_function_with_multiprocess import call_function_with_multiprocess
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .establish_path import establish_path
from .mf_consensus_cluster_dataframe import mf_consensus_cluster_dataframe
from .plot_heat_map import plot_heat_map
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED


def mf_consensus_cluster_dataframe_with_ks(
    dataframe,
    ks,
    directory_path,
    mf_function="nmf_with_sklearn",
    n_job=1,
    n_clustering=10,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_w=True,
    plot_h=True,
    plot_dataframe=True,
):

    k_directory_paths = tuple(join(directory_path, str(k)) for k in ks)

    for k_directory_path in k_directory_paths:

        establish_path(k_directory_path, "directory")

    k_return = {}

    for (
        k,
        (
            w_0,
            h_0,
            e_0,
            w_element_cluster,
            w_element_cluster_ccc,
            h_element_cluster,
            h_element_cluster_ccc,
        ),
    ) in zip(
        ks,
        call_function_with_multiprocess(
            mf_consensus_cluster_dataframe,
            (
                (
                    dataframe,
                    ks[i],
                    k_directory_paths[i],
                    mf_function,
                    n_clustering,
                    n_iteration,
                    random_seed,
                    linkage_method,
                    plot_w,
                    plot_h,
                    plot_dataframe,
                )
                for i in range(len(ks))
            ),
            n_job=n_job,
        ),
    ):

        k_return["K{}".format(k)] = {
            "w": w_0,
            "h": h_0,
            "e": e_0,
            "w_element_cluster": w_element_cluster,
            "w_element_cluster.ccc": w_element_cluster_ccc,
            "h_element_cluster": h_element_cluster,
            "h_element_cluster.ccc": h_element_cluster_ccc,
        }

    keys = Index(("K{}".format(k) for k in ks), name="K")

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "MF"},
                "xaxis": {"title": {"text": "K"}},
                "yaxis": {"title": {"text": "Error"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "x": ks,
                    "y": tuple(k_return[key]["e"] for key in keys),
                }
            ],
        },
        join(directory_path, "mf_error.html"),
    )

    w_element_cluster_ccc = tuple(
        k_return[key]["w_element_cluster.ccc"] for key in keys
    )

    h_element_cluster_ccc = tuple(
        k_return[key]["h_element_cluster.ccc"] for key in keys
    )

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "MFCC"},
                "xaxis": {"title": "K"},
                "yaxis": {"title": {"text": "Cophenetic Correlation Coefficient"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "name": "Mean",
                    "x": ks,
                    "y": (
                        asarray(w_element_cluster_ccc) + asarray(h_element_cluster_ccc)
                    )
                    / 2,
                },
                {"type": "scatter", "name": "W", "x": ks, "y": w_element_cluster_ccc},
                {"type": "scatter", "name": "H", "x": ks, "y": h_element_cluster_ccc},
            ],
        },
        join(directory_path, "ccc.html"),
    )

    for w_or_h, k_x_element in (
        (
            "w",
            DataFrame(
                [k_return[key]["w_element_cluster"] for key in keys],
                index=keys,
                columns=w_0.index,
            ),
        ),
        (
            "h",
            DataFrame(
                [k_return[key]["h_element_cluster"] for key in keys],
                index=keys,
                columns=h_0.columns,
            ),
        ),
    ):

        k_x_element.to_csv(
            join(directory_path, "k_x_{}_element.tsv".format(w_or_h)), sep="\t"
        )

        if plot_dataframe:

            plot_heat_map(
                DataFrame(sort(k_x_element.values, axis=1), index=keys),
                colorscale=DATA_TYPE_COLORSCALE["categorical"],
                layout={"title": {"text": "MFCC {}".format(w_or_h.title())}},
                html_file_path=join(
                    directory_path,
                    "k_x_{}_element.cluster_distribution.html".format(w_or_h),
                ),
            )

    return k_return
