from math import ceil
from multiprocessing import Pool

from numpy import apply_along_axis, asarray, concatenate, full, isnan, nan, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame, Series, unique

from .cluster import cluster
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .get_moe import get_moe
from .get_p_values_and_q_values import get_p_values_and_q_values
from .ignore_nan_and_function_1 import ignore_nan_and_function_1
from .ignore_nan_and_function_2 import ignore_nan_and_function_2
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
    directory_path=None,
):

    intersection = vector.index & matrix.columns

    print(
        "vector.index ({}) and matrix.columns ({}) share {}.".format(
            vector.index.size, matrix.columns.size, intersection.size
        )
    )

    vector = vector[intersection]

    if vector_ascending is not None:

        vector.sort_values(ascending=vector_ascending, inplace=True)

    matrix = matrix[vector.index]

    if statistics is None:

        statistics = DataFrame(
            index=matrix.index, columns=("Score", "0.95 MoE", "P-Value", "Q-Value")
        )

        n_matrix_row, n_matrix_column = matrix.shape

        n_job = min(n_matrix_row, n_job)

        print("Computing statistics using {} process...".format(n_job))

        pool = Pool(n_job)

        print("\tScore with {}...".format(function.__name__))

        seed(seed=random_seed)

        vector_ = vector.values

        statistics["Score"] = asarray(
            pool.starmap(
                ignore_nan_and_function_2,
                ((vector_, matrix_row, function) for matrix_row in matrix.values),
            )
        )

        if 0 < n_sampling:

            print("\t0.95 MoE with {} sampling...".format(n_sampling))

            row_x_sampling = full((n_matrix_row, n_sampling), nan)

            n_column_to_sample = ceil(0.632 * n_matrix_column)

            for sampling_index in range(n_sampling):

                columns = choice(n_matrix_column, size=n_column_to_sample)

                vector_ = vector.values[columns]

                row_x_sampling[:, sampling_index] = pool.starmap(
                    ignore_nan_and_function_2,
                    (
                        (vector_, matrix_row, function)
                        for matrix_row in matrix.values[:, columns]
                    ),
                )

            statistics["0.95 MoE"] = apply_along_axis(
                lambda scores: get_moe(scores[~isnan(scores)]), 1, row_x_sampling,
            )

        if 0 < n_permutation:

            print("\tP-Value and Q-Value with {} permutation...".format(n_sampling))

            row_x_permutation = full((n_matrix_row, n_permutation), nan)

            vector_ = vector.values.copy()

            for permuting_index in range(n_permutation):

                shuffle(vector_)

                row_x_permutation[:, permuting_index] = pool.starmap(
                    ignore_nan_and_function_2,
                    ((vector_, matrix_row, function) for matrix_row in matrix.values),
                )

            statistics["P-Value"], statistics["Q-Value"] = get_p_values_and_q_values(
                statistics["Score"].values, row_x_permutation.flatten(), "<>"
            )

        pool.terminate()

    else:

        statistics = statistics.reindex(row_name=matrix.index)

    statistics.sort_values("Score", ascending=score_ascending, inplace=True)

    if directory_path is not None:

        tsv_file_path = "{}/function_heat_map.tsv".format(directory_path)

        statistics.to_csv(tsv_file_path, sep="\t")

    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Statistics"},
                    "xaxis": {"title": {"text": "Rank"}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "name": name,
                        "x": values.index,
                        "y": values.values,
                    }
                    for name, values in statistics.items()
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

        matrix_ = matrix.loc[statistics_.index]

        if vector_data_type == "continuous":

            vector_ = Series(
                ignore_nan_and_function_1(
                    vector_.values, normalize, "-0-", update=True
                ),
                name=vector_.name,
                index=vector_.index,
            ).clip(lower=-plot_std, upper=plot_std)

        if matrix_data_type == "continuous":

            matrix_ = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1,
                    1,
                    matrix_.values,
                    normalize,
                    "-0-",
                    update=True,
                ),
                index=matrix_.index,
                columns=matrix_.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        if (
            not vector_.isna().any()
            and isinstance(vector_ascending, bool)
            and (1 < vector_.value_counts()).all()
        ):

            categories = vector_.values

            matrix_values_t = matrix_.values.T

            leave_index = []

            for category in unique(categories):

                row_name = where(categories == category)[0]

                leave_index.append(row_name[cluster(matrix_values_t[row_name])[0]])

            matrix_ = matrix_.iloc[:, concatenate(leave_index)]

            vector_ = vector_[matrix_.columns]

        n_row = 1 + 1 + matrix_.shape[0]

        fraction_row = 1 / n_row

        layout = {
            "height": max(480, 24 * n_row),
            "width": 800,
            "margin": {"l": 200, "r": 200},
            "title": {"x": 0.5},
            "yaxis": {"domain": (0, 1 - 2 * fraction_row), "showticklabels": False},
            "yaxis2": {"domain": (1 - fraction_row, 1), "showticklabels": False},
            "annotations": [],
        }

        annotation_ = {
            "xref": "paper",
            "yref": "paper",
            "yanchor": "middle",
            "font": {"size": 10},
            "showarrow": False,
        }

        layout["annotations"].append(
            {
                "x": 0,
                "y": 1 - fraction_row / 2,
                "xanchor": "right",
                "text": "<b>{}</b>".format(vector_.name),
                **annotation_,
            }
        )

        def get_x(x_index):

            return 1.1 + x_index / 6.4

        y = 1 - fraction_row / 2

        for x_index, str_ in enumerate(("Score (\u0394)", "P-Value", "Q-Value")):

            layout["annotations"].append(
                {
                    "x": get_x(x_index),
                    "y": y,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(str_),
                    **annotation_,
                }
            )

        y -= 2 * fraction_row

        for row_name, (score, moe, p_value, q_value) in statistics_.iterrows():

            layout["annotations"].append(
                {"x": 0, "y": y, "xanchor": "right", "text": row_name, **annotation_}
            )

            for x_index, str_ in enumerate(
                (
                    "{:.2f} ({:.2f})".format(score, moe),
                    "{:.2e}".format(p_value),
                    "{:.2e}".format(q_value),
                )
            ):

                layout["annotations"].append(
                    {
                        "x": get_x(x_index),
                        "y": y,
                        "xanchor": "center",
                        "text": str_,
                        **annotation_,
                    }
                )

            y -= fraction_row

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = tsv_file_path.repalce(".tsv", ".html")

        heatmap_trace_ = {
            "type": "heatmap",
            "showscale": False,
        }

        plot_plotly(
            {
                "layout": layout,
                "data": [
                    {
                        "yaxis": "y2",
                        "x": vector_.index,
                        "z": vector_.to_frame().T,
                        "colorscale": DATA_TYPE_COLORSCALE[vector_data_type],
                        **heatmap_trace_,
                    },
                    {
                        "yaxis": "y",
                        "x": matrix_.columns,
                        "y": matrix_.index[::-1],
                        "z": matrix_.iloc[::-1],
                        "colorscale": DATA_TYPE_COLORSCALE[matrix_data_type],
                        **heatmap_trace_,
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return statistics
