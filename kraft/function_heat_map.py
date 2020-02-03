from math import ceil
from multiprocessing import Pool

from numpy import apply_along_axis, asarray, full, isnan, nan
from numpy.random import choice, seed, shuffle
from pandas import DataFrame, Series

from .compute_margin_of_error import compute_margin_of_error
from .compute_p_values_and_q_values import compute_p_values_and_q_values
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .function_ignoring_nan import function_ignoring_nan
from .get_clustering_index import get_clustering_index
from .is_sorted import is_sorted
from .merge_2_dicts import merge_2_dicts
from .normalize import normalize
from .plot_plotly import plot_plotly
from .RANDOM_SEED import RANDOM_SEED
from .select_index import select_index


def function_heat_map(
    vector,
    matrix,
    function,
    vector_ascending=True,
    statistics=None,
    n_job=1,
    random_seed=RANDOM_SEED,
    n_sampling=10,
    n_permutation=10,
    score_ascending=False,
    plot=True,
    n_extreme=8,
    fraction_extreme=None,
    vector_data_type="continuous",
    matrix_data_type="continuous",
    plot_std=nan,
    cluster_within_category=True,
    layout=None,
    file_path_prefix=None,
):

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

    if statistics is None:

        statistics = DataFrame(
            index=matrix.index,
            columns=(
                "Score",
                "0.95 Margin of Error",
                "P-Value",
                "False Discovery Rate",
            ),
        )

        n_matrix_row, n_matrix_column = matrix.shape

        print("Computing statistics using {} process...".format(n_job))

        n_job = min(n_matrix_row, n_job)

        pool = Pool(n_job)

        print("Scoring...")

        seed(seed=random_seed)

        vector_ = vector.values

        statistics["Score"] = asarray(
            pool.starmap(
                function_ignoring_nan,
                ((vector_, matrix_row, function) for matrix_row in matrix.values),
            )
        )

        if 0 < n_sampling:

            print(
                "Computing 0.95 margin of error with {} sampling...".format(n_sampling)
            )

            row_x_sampling = full((n_matrix_row, n_sampling), nan)

            n_column_to_sample = ceil(0.632 * n_matrix_column)

            for sampling_index in range(n_sampling):

                columns = choice(n_matrix_column, size=n_column_to_sample)

                vector_ = vector.values[columns]

                row_x_sampling[:, sampling_index] = pool.starmap(
                    function_ignoring_nan,
                    (
                        (vector_, matrix_row, function)
                        for matrix_row in matrix.values[:, columns]
                    ),
                )

            statistics["0.95 Margin of Error"] = apply_along_axis(
                lambda sampled_scores: compute_margin_of_error(
                    sampled_scores[~isnan(sampled_scores)]
                ),
                1,
                row_x_sampling,
            )

        if 0 < n_permutation:

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
                    function_ignoring_nan,
                    ((vector_, matrix_row, function) for matrix_row in matrix.values),
                )

            (
                statistics["P-Value"],
                statistics["False Discovery Rate"],
            ) = compute_p_values_and_q_values(
                statistics["Score"].values, row_x_permutation.flatten(), "<>"
            )

        pool.terminate()

    else:

        statistics = statistics.reindex(index=matrix.index)

    statistics.sort_values("Score", ascending=score_ascending, inplace=True)

    if file_path_prefix is not None:

        statistics.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    if plot:

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

        vector_ = vector.copy()

        statistics_ = statistics.copy()

        if n_extreme is not None or fraction_extreme is not None:

            statistics_ = statistics_.loc[
                select_index(
                    statistics_["Score"],
                    "<>",
                    n=n_extreme,
                    fraction=fraction_extreme,
                    plot=False,
                )
            ].sort_values("Score", ascending=score_ascending)

        dataframe_ = matrix.loc[statistics_.index]

        if vector_data_type == "continuous":

            vector_ = Series(
                normalize(vector_.values, "-0-"),
                name=vector_.name,
                index=vector_.index,
            ).clip(lower=-plot_std, upper=plot_std)

        if matrix_data_type == "continuous":

            dataframe_ = DataFrame(
                apply_along_axis(normalize, 1, dataframe_.values, "-0-"),
                index=dataframe_.index,
                columns=dataframe_.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        if (
            cluster_within_category
            and not vector_.isna().any()
            and is_sorted(vector_.values)
            and (1 < vector_.value_counts()).all()
        ):

            print("Clustering within category...")

            dataframe_ = dataframe_.iloc[
                :, get_clustering_index(dataframe_.values, 1, groups=vector_.values),
            ]

            vector_ = vector_[dataframe_.columns]

        n_row = 1 + 1 + dataframe_.shape[0]

        fraction_row = 1 / n_row

        layout_template = {
            "height": max(500, 25 * n_row),
            "width": 800,
            "margin": {"l": 200, "r": 200},
            "title": {"x": 0.5},
            "yaxis": {"domain": (0, 1 - 2 * fraction_row), "showticklabels": False},
            "yaxis2": {"domain": (1 - fraction_row, 1), "showticklabels": False},
            "annotations": [],
        }

        if layout is None:

            layout = layout_template

        else:

            layout = merge_2_dicts(layout_template, layout)

        heatmap_trace_template = {
            "type": "heatmap",
            "zmin": -plot_std,
            "zmax": plot_std,
            "showscale": False,
        }

        annotation_template = {
            "xref": "paper",
            "yref": "paper",
            "yanchor": "middle",
            "font": {"size": 10},
            "showarrow": False,
        }

        layout["annotations"].append(
            {
                "x": 0,
                "y": 1 - (fraction_row / 2),
                "xanchor": "right",
                "text": "<b>{}</b>".format(vector_.name),
                **annotation_template,
            }
        )

        def get_x(ix):

            return 1.1 + ix / 6.4

        y = 1 - (fraction_row / 2)

        for ix, str_ in enumerate(("Score(\u0394)", "P-Value", "FDR")):

            layout["annotations"].append(
                {
                    "x": get_x(ix),
                    "y": y,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(str_),
                    **annotation_template,
                }
            )

        y -= 2 * fraction_row

        for (
            index,
            (score, margin_of_error, p_value, false_discovery_rate),
        ) in statistics_.iterrows():

            layout["annotations"].append(
                {
                    "x": 0,
                    "y": y,
                    "xanchor": "right",
                    "text": index,
                    **annotation_template,
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
                        **annotation_template,
                    }
                )

            y -= fraction_row

        if file_path_prefix is None:

            html_file_path = None

        else:

            html_file_path = "{}.html".format(file_path_prefix)

        plot_plotly(
            {
                "layout": layout,
                "data": [
                    {
                        "yaxis": "y2",
                        "name": "Target",
                        "x": vector_.index,
                        "z": vector_.to_frame().T,
                        "colorscale": DATA_TYPE_COLORSCALE[vector_data_type],
                        **heatmap_trace_template,
                    },
                    {
                        "yaxis": "y",
                        "name": "Data",
                        "x": dataframe_.columns,
                        "y": dataframe_.index[::-1],
                        "z": dataframe_.iloc[::-1],
                        "colorscale": DATA_TYPE_COLORSCALE[matrix_data_type],
                        **heatmap_trace_template,
                    },
                ],
            },
            html_file_path,
        )

    return statistics
