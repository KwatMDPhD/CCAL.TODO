from math import ceil
from multiprocessing import Pool

from numpy import (
    asarray,
    argsort,
    concatenate,
    full,
    isnan,
    nan,
    unique,
    where,
)
from numpy.random import choice, seed, shuffle

from .clustering import cluster
from .CONSTANT import RANDOM_SEED, SAMPLE_FRACTION
from .dictionary import merge
from .plot import DATA_TYPE_TO_COLORSCALE, plot_plotly
from .significance import get_margin_of_error, get_p_values_and_q_values
from .array import (
    function_on_1_number_array_not_nan,
    function_on_2_number_array_not_nan,
    check_is_all_sorted,
    normalize,
    check_is_extreme,
    check_is_not_nan,
)
from .table import entangle, untangle

HEATMAP_BASE = {
    "type": "heatmap",
    "showscale": False,
}

LAYOUT_BASE = {
    "width": 800,
    "margin": {"l": 200, "r": 200},
    "title": {"x": 0.5},
}

ANNOTATION_BASE = {
    "xref": "paper",
    "yref": "paper",
    "yanchor": "middle",
    "font": {"size": 10},
    "showarrow": False,
}


def _annotate_se(text, y):

    return {
        "x": 0,
        "y": y,
        "xanchor": "right",
        "text": "<b>{}</b>".format(text),
        **ANNOTATION_BASE,
    }


def _get_x(score_index):

    return 1.08 + score_index / 6.4


def _annotate_table(axis_0_label_, score_matrix, y, row_height, add_score_header):

    annotation_ = []

    if add_score_header:

        for score_index, text in enumerate(("Score (\u0394)", "P-Value", "Q-Value")):

            annotation_.append(
                {
                    "x": _get_x(score_index),
                    "y": y,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION_BASE,
                }
            )

    y -= row_height

    for axis_0_index in axis_0_label_.size:

        annotation_.append(
            {
                "x": 0,
                "y": y,
                "xanchor": "right",
                "text": axis_0_label_[axis_0_index],
                **ANNOTATION_BASE,
            }
        )

        score, moe, p_value, q_value = score_matrix[axis_0_index]

        for score_index, text in enumerate(
            (
                "{:.2f} ({:.2f})".format(score, moe),
                "{:.2e}".format(p_value),
                "{:.2e}".format(q_value),
            )
        ):

            annotation_.append(
                {
                    "x": _get_x(score_index),
                    "y": y,
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION_BASE,
                }
            )

        y -= row_height

    return annotation_


def make(
    vector,
    matrix,
    axis_0_label_,
    axis_1_label_,
    function_or_statistic_dataframe,
    job_number=1,
    random_seed=RANDOM_SEED,
    sample_number=10,
    shuffle_number=10,
    plot=True,
    plot_number=8,
    vector_data_type="continuous",
    matrix_data_type="continuous",
    plot_standard_deviation=nan,
    title="Function Heat Map",
    directory_path=None,
):

    if callable(function_or_statistic_dataframe):

        function = function_or_statistic_dataframe

        axis_0_size, axis_1_size = matrix.shape

        pool = Pool(job_number)

        seed(seed=random_seed)

        print("Score (function={})...".format(function.__name__))

        score_vector = asarray(
            pool.starmap(
                function_on_2_number_array_not_nan,
                ((vector, row_vector, function) for row_vector in matrix),
            )
        )

        if 0 < sample_number:

            print("Margin of Error (0.95) (sample_number={})...".format(sample_number))

            row_x_sample_matrix = full((axis_0_size, sample_number), nan)

            sample_number = ceil(axis_1_size * SAMPLE_FRACTION)

            for sample_index in range(sample_number):

                index_ = choice(axis_1_size, size=sample_number)

                vector_sample = vector[index_]

                row_x_sample_matrix[:, sample_index] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    (
                        (vector_sample, row_vector_sample, function)
                        for row_vector_sample in matrix[:, index_]
                    ),
                )

            margin_of_error_vector = asarray(
                tuple(
                    function_on_1_number_array_not_nan(row_vector, get_margin_of_error)
                    for row_vector in row_x_sample_matrix
                )
            )

        if 0 < shuffle_number:

            print("P-Value and Q-Value (shuffle_number={})...".format(shuffle_number))

            row_x_shuffle_matrix = full((axis_0_size, shuffle_number), nan)

            vector_shuffle = vector.copy()

            for shuffle_index in range(shuffle_number):

                shuffle(vector_shuffle)

                row_x_shuffle_matrix[:, shuffle_index] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    ((vector_shuffle, row_vector, function) for row_vector in matrix),
                )

            p_value_vector, q_value_vector = get_p_values_and_q_values(
                score_vector, row_x_shuffle_matrix.ravel(), "<>"
            )

        pool.terminate()

        statistic_matrix = asarray(
            (score_vector, margin_of_error_vector, p_value_vector, q_value_vector)
        ).T

    else:

        statistic_matrix = untangle(function_or_statistic_dataframe)[0]

    index_ = argsort(statistic_matrix[:, 0])

    statistic_matrix = statistic_matrix[index_]

    matrix = matrix[index_]

    axis_0_label_ = axis_0_label_[index_]

    statistic_label_ = asarray(("Score", "Margin of Error", "P-Value", "Q-Value"))

    if plot:

        plot_plotly(
            {
                "data": [
                    {
                        "name": statistic_label_[statistic_index],
                        "x": statistic_matrix[:, statistic_index],
                        "y": axis_0_label_,
                    }
                    for statistic_index in range(statistic_label_.size)
                ],
                "layout": {
                    "title": {"text": title},
                    "xaxis": {"title": {"text": "Statistic"}},
                    "yaxis": {"title": {"text": "Rank"}},
                },
            },
        )

        if 0 < plot_number:

            is_ = check_is_extreme(matrix[:, 0], "<>", number=plot_number)

            statistic_matrix = statistic_matrix[is_]

            matrix = matrix[is_]

            axis_0_label_ = axis_0_label_[is_]

        if vector_data_type == "continuous":

            print("Normalizing vector heat...")

            vector = normalize(vector, "-0-").clip(
                min=-plot_standard_deviation, max=plot_standard_deviation
            )

            vector_minimum = -plot_standard_deviation

            vector_maximum = plot_standard_deviation

        else:

            vector_minimum = None

            vector_maximum = None

        if matrix_data_type == "continuous":

            print("Normalizing matrix heat...")

            matrix = matrix.copy()

            for row_index in range(matrix.shape[0]):

                matrix[row_index] = normalize(matrix[row_index], "-0-").clip(
                    min=-plot_standard_deviation, max=plot_standard_deviation
                )

            matrix_minimum = -plot_standard_deviation

            matrix_maximum = plot_standard_deviation

        else:

            matrix_minimum = None

            matrix_maximum = None

        if (
            check_is_not_nan(vector).all()
            and check_is_all_sorted(vector)
            and (1 < unique(vector, return_counts=True)[1]).all()
        ):

            print("Clustering...")

            leaf_index_ = []

            for number in unique(vector):

                index_ = where(vector == number)[0]

                leaf_index_.append(index_[cluster(matrix.T[index_])[0]])

            leaf_index_ = concatenate(leaf_index_)

            vector = vector[leaf_index_]

            matrix = matrix[:, leaf_index_]

            axis_1_label_[leaf_index_]

        axis_0_size = 1 + 1 + df.shape[0]

        row_fraction = 1 / axis_0_size

        layout = merge(
            {
                "height": max(480, 24 * axis_0_size),
                "yaxis": {"domain": (0, 1 - row_fraction * 2), "showticklabels": False},
                "yaxis2": {"domain": (1 - row_fraction, 1), "showticklabels": False},
                "title": {"text": title},
                "annotations": [_annotate_se(se, 1 - row_fraction / 2)],
            },
            LAYOUT_BASE,
        )

        layout["annotations"] += _annotate_table(
            statistic_dataframe_plot, 1 - row_fraction / 2 * 3, row_fraction, True
        )

        if directory_path is None:

            file_path = None

        else:

            file_path = file_path.replace(".tsv", ".html")

        plot_plotly(
            {
                "layout": layout,
                "data": [
                    {
                        "yaxis": "y2",
                        "x": se.index.to_numpy(),
                        "z": vector.reshape((1, -1)),
                        "zmin": vector_minimum,
                        "zmax": vector_maximum,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[se_data_type],
                        **HEATMAP_BASE,
                    },
                    {
                        "yaxis": "y",
                        "x": df.columns.to_numpy(),
                        "y": df.index.to_numpy()[::-1],
                        "z": matrix[::-1],
                        "zmin": matrix_minimum,
                        "zmax": matrix_maximum,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[table_data_type],
                        **HEATMAP_BASE,
                    },
                ],
            },
            file_path=file_path,
        )

    return statistic_dataframe


def summarize(
    se,
    table_dicts,
    scores,
    plot_only_shared=False,
    se_ascending=True,
    se_data_type="continuous",
    plot_std=nan,
    title="Function Heat Map Summary",
    file_path=None,
):

    if plot_only_shared:

        for dict_ in table_dicts.values():

            se = se.loc[se.index & dict_["df"].columns]

    if se_ascending is not None:

        se.sort_values(ascending=se_ascending, inplace=True)

    if se_data_type == "continuous" and 1 < get_not_nan_unique(se.to_numpy()).size:

        print("Normalizing heat...")

        se = series_normalize(se, "-0-").clip(lower=-plot_std, upper=plot_std)

        se_min = -plot_std

        se_max = plot_std

    else:

        se_min = None

        se_max = None

    space_number = 2

    row_number = 1

    for dict_ in table_dicts.values():

        row_number += space_number

        row_number += dict_["df"].shape[0]

    row_fraction = 1 / row_number

    layout = merge(
        {
            "height": max(480, 24 * row_number),
            "title": {"text": title},
            "annotations": [_annotate_se(se, 1 - row_fraction / 2)],
        },
        LAYOUT_BASE,
    )

    yaxis = "yaxis{}".format(len(table_dicts) + 1)

    domain = 1 - row_fraction, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "x": se.index.to_numpy(),
            "z": se.to_numpy().reshape((1, -1)),
            "zmin": se_min,
            "zmax": se_max,
            "colorscale": DATA_TYPE_TO_COLORSCALE[se_data_type],
            **HEATMAP_BASE,
        }
    ]

    for i, (name, dict_) in enumerate(table_dicts.items()):

        df = dict_["df"].reindex(labels=se.index, axis=1)

        scores_ = scores[name].reindex(labels=df.index)

        if "emphasis" in dict_:

            score_ascending = dict_["emphasis"] == "-"

        else:

            score_ascending = False

        scores_.sort_values("Score", ascending=score_ascending, inplace=True)

        df = df.loc[scores_.index, :]

        if dict_["data_type"] == "continuous" and all(
            1 < get_not_nan_unique(row).size for row in df.dropna(how="all").to_numpy()
        ):

            print("Normalizing heat...")

            df = dataframe_normalize(df, 1, "-0-").clip(lower=-plot_std, upper=plot_std)

            table_min = -plot_std

            table_max = plot_std

        else:

            table_min = None

            table_max = None

        yaxis = "yaxis{}".format(len(table_dicts) - i)

        domain = (
            max(0, domain[0] - row_fraction * (space_number + df.shape[0])),
            domain[0] - row_fraction * space_number,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "x": df.columns.to_numpy(),
                "y": df.index.to_numpy()[::-1],
                "z": df.to_numpy()[::-1],
                "zmin": table_min,
                "zmax": table_max,
                "colorscale": DATA_TYPE_TO_COLORSCALE[dict_["data_type"]],
                **HEATMAP_BASE,
            }
        )

        y = domain[1] + row_fraction / 2

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": y,
                "xanchor": "center",
                "text": "<b>{}</b>".format(name),
                **ANNOTATION_BASE,
            }
        )

        layout["annotations"] += _annotate_table(scores_, y, row_fraction, i == 0)

    plot_plotly({"layout": layout, "data": data}, file_path=file_path)
