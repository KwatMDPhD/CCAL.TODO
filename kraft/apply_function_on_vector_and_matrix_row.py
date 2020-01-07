from math import ceil
from multiprocessing import Pool

from numpy import apply_along_axis, asarray, full, nan
from numpy.random import choice, seed, shuffle
from pandas import DataFrame, Series

from .cluster_matrix import cluster_matrix
from .compute_margin_of_error import compute_margin_of_error
from .compute_p_values_and_false_discovery_rates import (
    compute_p_values_and_false_discovery_rates,
)
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .ignore_nan_and_apply_function_on_2_vectors import (
    ignore_nan_and_apply_function_on_2_vectors,
)
from .is_sorted_array import is_sorted_array
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .normalize_array import normalize_array
from .plot_plotly import plot_plotly
from .RANDOM_SEED import RANDOM_SEED
from .select_series_index import select_series_index


def apply_function_on_vector_and_matrix_row(
    vector,
    matrix,
    function,
    vector_ascending=True,
    n_job=1,
    random_seed=RANDOM_SEED,
    n_sampling=10,
    n_permutation=10,
    statistics=None,
    score_ascending=False,
    n_extreme=8,
    fraction_extreme=None,
    plot=True,
    series_data_type="continuous",
    dataframe_data_type="continuous",
    plot_std=nan,
    cluster_within_category=True,
    layout=None,
    file_path_prefix=None,
):

    #
    intersection = vector.index & matrix.columns

    print(
        "vector.index ({}) and matrix.columns ({}) share {}.".format(
            vector.index.size, matrix.columns.size, intersection.size,
        )
    )

    vector = vector[intersection]

    if vector_ascending is not None:

        vector.sort_values(ascending=vector_ascending, inplace=True)

    matrix = matrix[vector.index]

    #
    if statistics is None:

        #
        statistics = DataFrame(index=matrix.index)

        #
        seed(seed=random_seed)

        #
        n_matrix_row, n_matrix_column = matrix.shape

        #
        print("Computing statistics using {} process...".format(n_job))

        n_job = min(n_matrix_row, n_job)

        pool = Pool(n_job)

        #
        print("Scoring...")

        vector_ = vector.values

        statistics["Score"] = asarray(
            pool.starmap(
                ignore_nan_and_apply_function_on_2_vectors,
                ((vector_, matrix_row, function) for matrix_row in matrix.values),
            )
        )

        #
        print("Computing 0.95 margin of error with {} sampling...".format(n_sampling))

        row_x_sampling = full((n_matrix_row, n_sampling), nan)

        n_column_to_sample = ceil(0.632 * n_matrix_column)

        for sampling_index in range(n_sampling):

            columns = choice(n_matrix_column, size=n_column_to_sample)

            vector_ = vector.values[columns]

            row_x_sampling[:, sampling_index] = pool.starmap(
                ignore_nan_and_apply_function_on_2_vectors,
                (
                    (vector_, matrix_row, function)
                    for matrix_row in matrix.values[:, columns]
                ),
            )

        statistics["0.95 Margin of Error"] = apply_along_axis(
            lambda sampled_scores: compute_margin_of_error(
                sampled_scores[sampled_scores != nan]
            ),
            1,
            row_x_sampling,
        )

        #
        print(
            "Computing p-value and false discovery rate with {} permutation...".format(
                n_sampling
            )
        )

        row_x_permutation = full((n_matrix_row, n_permutation), nan)

        vector_ = vector.values.copy()

        for permuting_index in range(n_permutation):

            shuffle(vector_)

            row_x_permutation[:, permuting_index] = pool.starmap(
                ignore_nan_and_apply_function_on_2_vectors,
                ((vector_, matrix_row, function) for matrix_row in matrix.values),
            )

        (
            statistics["P-Value"],
            statistics["False Discovery Rate"],
        ) = compute_p_values_and_false_discovery_rates(
            statistics["Score"].values, row_x_permutation.flatten(), "<>"
        )

        #
        pool.terminate()

    else:

        statistics = statistics.reindex(index=matrix.index)

    #
    statistics.sort_values("Score", ascending=score_ascending, inplace=True)

    if file_path_prefix is not None:

        statistics.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    if not plot:

        return statistics

    #
    print("Plotting...")

    plot_plotly(
        {
            "layout": {
                "title": {"text": "Statistics"},
                "xaxis": {"title": {"text": "Rank"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "name": statistics_name,
                    "x": statistics_column.index,
                    "y": statistics_column.values,
                }
                for statistics_name, statistics_column in statistics.items()
            ],
        },
    )

    #
    vector_plot = vector.copy()

    statistics_plot = statistics.copy()

    if n_extreme is not None or fraction_extreme is not None:

        statistics_plot = statistics_plot.loc[
            select_series_index(
                statistics_plot["Score"],
                "<>",
                n=n_extreme,
                fraction=fraction_extreme,
                plot=False,
            )
        ].sort_values("Score", ascending=score_ascending)

    dataframe_plot = matrix.loc[statistics_plot.index]

    #
    if series_data_type == "continuous":

        vector_plot = Series(
            normalize_array(vector_plot.values, "-0-"),
            name=vector_plot.name,
            index=vector_plot.index,
        ).clip(lower=-plot_std, upper=plot_std)

    if dataframe_data_type == "continuous":

        dataframe_plot = DataFrame(
            apply_along_axis(normalize_array, 1, dataframe_plot.values, "-0-"),
            index=dataframe_plot.index,
            columns=dataframe_plot.columns,
        ).clip(lower=-plot_std, upper=plot_std)

    #
    if (
        cluster_within_category
        and not vector_plot.isna().any()
        and is_sorted_array(vector_plot.values)
        and (1 < vector_plot.value_counts()).all()
    ):

        print("Clustering within category...")

        dataframe_plot = dataframe_plot.iloc[
            :, cluster_matrix(dataframe_plot.values, 1, groups=vector_plot.values),
        ]

        vector_plot = vector_plot[dataframe_plot.columns]

    #
    n_row = 1 + 1 + dataframe_plot.shape[0]

    row_fraction = 1 / n_row

    layout_template = {
        "height": max(500, 25 * n_row),
        "width": 800,
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        # "xaxis": {"showticklabels": False},
        "yaxis": {"domain": (0, 1 - 2 * row_fraction), "showticklabels": False},
        "yaxis2": {"domain": (1 - row_fraction, 1), "showticklabels": False},
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

    #
    heatmap_trace_template = {
        "type": "heatmap",
        "zmin": -plot_std,
        "zmax": plot_std,
        "showscale": False,
    }

    data = [
        {
            "yaxis": "y2",
            "name": "Target",
            "x": vector_plot.index,
            "z": vector_plot.to_frame().T,
            "colorscale": DATA_TYPE_COLORSCALE[series_data_type],
            **heatmap_trace_template,
        },
        {
            "yaxis": "y",
            "name": "Data",
            "x": dataframe_plot.columns,
            "y": dataframe_plot.index[::-1],
            "z": dataframe_plot.iloc[::-1],
            "colorscale": DATA_TYPE_COLORSCALE[dataframe_data_type],
            **heatmap_trace_template,
        },
    ]

    #
    layout_annotation_template = {
        "xref": "paper",
        "yref": "paper",
        "yanchor": "middle",
        "font": {"size": 10},
        "showarrow": False,
    }

    #
    layout["annotations"].append(
        {
            "x": 0,
            "y": 1 - (row_fraction / 2),
            "xanchor": "right",
            "text": "<b>{}</b>".format(vector.name),
            **layout_annotation_template,
        }
    )

    def get_x(ix):

        return 1.1 + ix / 6.4

    y = 1 - (row_fraction / 2)

    for ix, str_ in enumerate(("Score(\u0394)", "P-Value", "FDR")):

        layout["annotations"].append(
            {
                "x": get_x(ix),
                "y": y,
                "xanchor": "center",
                "text": "<b>{}</b>".format(str_),
                **layout_annotation_template,
            }
        )

    #
    y -= 2 * row_fraction

    for (
        index,
        (score, margin_of_error, p_value, false_discovery_rate),
    ) in statistics_plot.iterrows():

        layout["annotations"].append(
            {
                "x": 0,
                "y": y,
                "xanchor": "right",
                "text": index,
                **layout_annotation_template,
            }
        )

        for ix, str_ in enumerate(
            (
                "{:.2f}({:.2f})".format(score, margin_of_error),
                "{:.2e}".format(p_value),
                "{:.2e}".format(false_discovery_rate),
            )
        ):

            layout["annotations"].append(
                {
                    "x": get_x(ix),
                    "y": y,
                    "xanchor": "center",
                    "text": str_,
                    **layout_annotation_template,
                }
            )

        y -= row_fraction

    #
    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.html".format(file_path_prefix)

    plot_plotly({"layout": layout, "data": data}, html_file_path)

    return statistics
