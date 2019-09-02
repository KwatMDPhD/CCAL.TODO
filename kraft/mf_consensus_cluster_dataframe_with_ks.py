from os.path import join

from numpy import asarray, sort
from pandas import DataFrame, Index

from .call_function_with_multiprocess import call_function_with_multiprocess
from .establish_path import establish_path
from .mf_consensus_cluster_dataframe import mf_consensus_cluster_dataframe
from .plot_heat_map import plot_heat_map
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED


def mf_consensus_cluster_dataframe_with_ks(
    dataframe,
    ks,
    mf_function="nmf_by_sklearn",
    n_job=1,
    n_clustering=10,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_w=True,
    plot_h=True,
    plot_dataframe=True,
    directory_path=None,
):

    if directory_path is None:

        k_directory_paths = tuple(None for k in ks)

    else:

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
            w_element_cluster__ccc,
            h_element_cluster,
            h_element_cluster__ccc,
        ),
    ) in zip(
        ks,
        call_function_with_multiprocess(
            mf_consensus_cluster_dataframe,
            (
                (
                    dataframe,
                    k,
                    mf_function,
                    n_clustering,
                    n_iteration,
                    random_seed,
                    linkage_method,
                    plot_w,
                    plot_h,
                    plot_dataframe,
                    k_directory_path,
                )
                for k, k_directory_path in zip(ks, k_directory_paths)
            ),
            n_job=n_job,
        ),
    ):

        k_return[f"K{k}"] = {
            "w": w_0,
            "h": h_0,
            "e": e_0,
            "w_element_cluster": w_element_cluster,
            "w_element_cluster.ccc": w_element_cluster__ccc,
            "h_element_cluster": h_element_cluster,
            "h_element_cluster.ccc": h_element_cluster__ccc,
        }

    keys = Index((f"K{k}" for k in ks), name="K")

    file_name = "mf_error.html"

    if directory_path is None:

        html_file_path = None

    else:

        html_file_path = join(directory_path, file_name)

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "MF Error"},
                "xaxis": {"title": {"text": "K"}},
                "yaxis": {"title": {"text": "Error"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "x": ks,
                    "y": tuple(k_return[key]["e"] for key in keys),
                    "mode": "lines+markers",
                    "marker": {"color": "#ff1968"},
                }
            ],
        },
        html_file_path,
    )

    w_element_cluster__ccc = asarray(
        tuple(k_return[key]["w_element_cluster.ccc"] for key in keys)
    )

    h_element_cluster__ccc = asarray(
        tuple(k_return[key]["h_element_cluster.ccc"] for key in keys)
    )

    file_name = "ccc.html"

    if directory_path is None:

        html_file_path = None

    else:

        html_file_path = join(directory_path, file_name)

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "MFCC Cophenetic Correlation Coefficient"},
                "xaxis": {"title": "K"},
                "yaxis": {"title": "CCC"},
            },
            "data": [
                {
                    "type": "scatter",
                    "name": "W Element Cluster CCC",
                    "x": ks,
                    "y": w_element_cluster__ccc,
                    "mode": "lines+markers",
                    "marker": {"color": "#9017e6"},
                },
                {
                    "type": "scatter",
                    "name": "H Element Cluster CCC",
                    "x": ks,
                    "y": h_element_cluster__ccc,
                    "mode": "lines+markers",
                    "marker": {"color": "#4e40d8"},
                },
                {
                    "type": "scatter",
                    "name": "W H Mean",
                    "x": ks,
                    "y": (w_element_cluster__ccc + h_element_cluster__ccc) / 2,
                    "mode": "lines+markers",
                    "marker": {"color": "#20d9ba"},
                },
            ],
        },
        html_file_path,
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

        if directory_path is not None:

            k_x_element.to_csv(
                join(directory_path, f"k_x_{w_or_h}_element.tsv"), sep="\t"
            )

        if plot_dataframe:

            file_name = f"k_x_{w_or_h}_element.cluster_distribution.html"

            if directory_path is None:

                html_file_path = None

            else:

                html_file_path = join(directory_path, file_name)

            plot_heat_map(
                DataFrame(sort(k_x_element.values, axis=1), index=keys),
                title_text=f"MFCC {w_or_h.title()} Cluster Distribution",
                xaxis_title_text=f"{w_or_h.title()} Element",
                yaxis_title_text=k_x_element.index.name,
                html_file_path=html_file_path,
            )

    return k_return
