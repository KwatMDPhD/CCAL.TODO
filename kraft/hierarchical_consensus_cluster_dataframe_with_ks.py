from os.path import join

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
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED


def hierarchical_consensus_cluster_dataframe_with_ks(
    dataframe,
    ks,
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

    k_directory_paths = tuple(join(directory_path, str(k)) for k in ks)

    for k_directory_path in k_directory_paths:

        establish_path(k_directory_path, "directory")

    if element_x_element_distance is None:

        if axis == 1:

            dataframe = dataframe.T

        print(
            "Computing element_x_element_distance with {}...".format(distance_function)
        )

        element_x_element_distance = DataFrame(
            squareform(pdist(dataframe.values, distance_function)),
            index=dataframe.index,
            columns=dataframe.index,
        )

        element_x_element_distance.to_csv(
            join(directory_path, "distance.element_x_element.tsv"), sep="\t"
        )

        if axis == 1:

            dataframe = dataframe.T

    k_return = {}

    for k, (element_cluster, element_cluster_ccc) in zip(
        ks,
        call_function_with_multiprocess(
            hierarchical_consensus_cluster_dataframe,
            (
                (
                    dataframe,
                    ks[i],
                    axis,
                    k_directory_paths[i],
                    element_x_element_distance,
                    None,
                    n_clustering,
                    random_seed,
                    linkage_method,
                    plot_dataframe,
                )
                for i in range(len(ks))
            ),
            n_job=n_job,
        ),
    ):

        k_return["K{}".format(k)] = {
            "element_cluster": element_cluster,
            "element_cluster.ccc": element_cluster_ccc,
        }

    keys = Index(("K{}".format(k) for k in ks), name="K")

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "HCC"},
                "xaxis": {"title": {"text": "K"}},
                "yaxis": {"title": {"text": "Cophenetic Correlation Coefficient"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "x": ks,
                    "y": tuple(k_return[key]["element_cluster.ccc"] for key in keys),
                }
            ],
        },
        join(directory_path, "ccc.html"),
    )

    k_x_element = concat([k_return[key]["element_cluster"] for key in keys], axis=1).T

    k_x_element.index = keys

    k_x_element.to_csv(join(directory_path, "k_x_element.tsv"), sep="\t")

    if plot_dataframe:

        plot_heat_map(
            DataFrame(
                sort(k_x_element.values, axis=1),
                index=k_x_element.index,
                columns=k_x_element.columns,
            ),
            colorscale=DATA_TYPE_COLORSCALE["categorical"],
            layout={"title": {"text": "HCC"}},
            html_file_path=join(
                directory_path, "k_x_element.cluster_distribution.html"
            ),
        )

    return k_return
