from numpy import asarray, sort
from pandas import DataFrame, Index

from .call_function_with_multiprocess import call_function_with_multiprocess
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .establish_path import establish_path
from .mf_consensus_cluster_dataframe import mf_consensus_cluster_dataframe
from .plot_heat_map import plot_heat_map
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED


def mf_consensus_cluster_dataframe_with_rs(
    dataframe,
    rs,
    directory_path,
    mf_function="nmf_with_sklearn",
    n_job=1,
    n_clustering=10,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_heat_map_=True,
    **plot_mf_keyword_arguments,
):

    r_directory_paths = tuple("{}/{}".format(directory_path, r) for r in rs)

    for r_directory_path in r_directory_paths:

        establish_path(r_directory_path, "directory")

    r_return = {}

    for (
        r,
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
        rs,
        call_function_with_multiprocess(
            mf_consensus_cluster_dataframe,
            (
                (
                    dataframe,
                    rs[i],
                    r_directory_paths[i],
                    mf_function,
                    n_clustering,
                    n_iteration,
                    random_seed,
                    linkage_method,
                    plot_heat_map_,
                )
                for i in range(len(rs))
            ),
            n_job=n_job,
        ),
    ):

        r_return["r{}".format(r)] = {
            "w": w_0,
            "h": h_0,
            "e": e_0,
            "w_element_cluster": w_element_cluster,
            "w_element_cluster_ccc": w_element_cluster_ccc,
            "h_element_cluster": h_element_cluster,
            "h_element_cluster_ccc": h_element_cluster_ccc,
        }

    keys = Index(("r{}".format(r) for r in rs), name="r")

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "MF"},
                "xaxis": {"title": {"text": "r"}},
                "yaxis": {"title": {"text": "Error"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "x": rs,
                    "y": tuple(r_return[key]["e"] for key in keys),
                }
            ],
        },
        "{}/mf_error.html".format(directory_path),
    )

    w_element_cluster_ccc = tuple(
        r_return[key]["w_element_cluster_ccc"] for key in keys
    )

    h_element_cluster_ccc = tuple(
        r_return[key]["h_element_cluster_ccc"] for key in keys
    )

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "MFCC"},
                "xaxis": {"title": "r"},
                "yaxis": {"title": {"text": "Cophenetic Correlation Coefficient"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "name": "Mean",
                    "x": rs,
                    "y": (
                        asarray(w_element_cluster_ccc) + asarray(h_element_cluster_ccc)
                    )
                    / 2,
                },
                {"type": "scatter", "name": "W", "x": rs, "y": w_element_cluster_ccc},
                {"type": "scatter", "name": "H", "x": rs, "y": h_element_cluster_ccc},
            ],
        },
        "{}/ccc.html".format(directory_path),
    )

    for w_or_h, r_x_element in (
        (
            "w",
            DataFrame(
                [r_return[key]["w_element_cluster"] for key in keys],
                index=keys,
                columns=w_0.index,
            ),
        ),
        (
            "h",
            DataFrame(
                [r_return[key]["h_element_cluster"] for key in keys],
                index=keys,
                columns=h_0.columns,
            ),
        ),
    ):

        r_x_element.to_csv(
            "{}/r_x_{}_element.tsv".format(directory_path, w_or_h), sep="\t"
        )

        if plot_heat_map_:

            plot_heat_map(
                DataFrame(sort(r_x_element.values, axis=1), index=keys),
                colorscale=DATA_TYPE_COLORSCALE["categorical"],
                layout={"title": {"text": "MFCC {}".format(w_or_h.title())}},
                html_file_path="{}/r_x_{}_element_cluster_distribution.html".format(
                    directory_path, w_or_h
                ),
            )

    return r_return
