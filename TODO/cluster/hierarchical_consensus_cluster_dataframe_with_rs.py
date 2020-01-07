from numpy import sort
from pandas import DataFrame, Index, concat
from scipy.spatial.distance import pdist, squareform

from .call_function_with_multiprocess import call_function_with_multiprocess
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .establish_path import establish_path
from .hierarchical_consensus_cluster_dataframe import (
    hierarchical_consensus_cluster_dataframe,
)
from .plot_heat_map import plot_heat_map
from .plot_plotly import plot_plotly
from .RANDOM_SEED import RANDOM_SEED


def hierarchical_consensus_cluster_dataframe_with_rs(
    dataframe,
    rs,
    axis,
    directory_path,
    element_x_element_distance=None,
    distance_function="correlation",
    n_job=1,
    n_clustering=10,
    random_seed=RANDOM_SEED,
    linkage_method="ward",
    plot_dataframe=True,
):

    r_directory_paths = tuple("{}/{}".format(directory_path, r) for r in rs)

    for r_directory_path in r_directory_paths:

        establish_path(r_directory_path, "directory")

    if element_x_element_distance is None:

        if axis == 1:

            dataframe = dataframe.T

        print("Computing element-element distance with {}...".format(distance_function))

        element_x_element_distance = DataFrame(
            squareform(pdist(dataframe.values, distance_function)),
            index=dataframe.index,
            columns=dataframe.index,
        )

        element_x_element_distance.to_csv(
            "{}/element_x_element_distance.tsv".format(directory_path), sep="\t"
        )

        if axis == 1:

            dataframe = dataframe.T

    r_return = {}

    for r, (element_cluster, element_cluster_ccc) in zip(
        rs,
        call_function_with_multiprocess(
            hierarchical_consensus_cluster_dataframe,
            (
                (
                    dataframe,
                    rs[i],
                    axis,
                    r_directory_paths[i],
                    element_x_element_distance,
                    None,
                    n_clustering,
                    random_seed,
                    linkage_method,
                    plot_dataframe,
                )
                for i in range(len(rs))
            ),
            n_job=n_job,
        ),
    ):

        r_return["r{}".format(r)] = {
            "element_cluster": element_cluster,
            "element_cluster_ccc": element_cluster_ccc,
        }

    keys = Index(("r{}".format(r) for r in rs), name="r")

    plot_plotly(
        {
            "layout": {
                "title": {"text": "HCC"},
                "xaxis": {"title": {"text": "r"}},
                "yaxis": {"title": {"text": "Cophenetic Correlation Coefficient"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "x": rs,
                    "y": tuple(r_return[key]["element_cluster_ccc"] for key in keys),
                }
            ],
        },
        "{}/ccc.html".format(directory_path),
    )

    r_x_element = concat([r_return[key]["element_cluster"] for key in keys], axis=1).T

    r_x_element.index = keys

    r_x_element.to_csv("{}/r_x_element.tsv".format(directory_path), sep="\t")

    if plot_dataframe:

        plot_heat_map(
            DataFrame(
                sort(r_x_element.values, axis=1),
                index=r_x_element.index,
                columns=r_x_element.columns,
            ),
            colorscale=DATA_TYPE_COLORSCALE["categorical"],
            layout={"title": {"text": "HCC"}},
            html_file_path="{}/r_x_element_cluster_distribution.html".format(
                directory_path
            ),
        )

    return r_return
