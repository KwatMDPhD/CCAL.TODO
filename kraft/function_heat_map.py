from multiprocessing import Pool

from numpy import argsort, asarray, concatenate, full, nan, unique, where
from numpy.random import choice, seed, shuffle

from .array import (
    check_is_all_sorted,
    check_is_extreme,
    check_is_not_nan,
    function_on_1_number_array_not_nan,
    function_on_2_number_array_not_nan,
    normalize,
)
from .clustering import cluster
from .CONSTANT import GOLDEN_FACTOR, RANDOM_SEED, SAMPLE_FRACTION
from .dictionary import merge
from .plot import DATA_TYPE_TO_COLORSCALE, plot_plotly
from .significance import get_margin_of_error, get_p_values_and_q_values
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


def _make_vector_annotation(text, y):

    return [
        {
            "x": 0,
            "y": y,
            "xanchor": "right",
            "text": "<b>{}</b>".format(text),
            **ANNOTATION_BASE,
        }
    ]


def _get_x(score_index):

    return 1.08 + score_index / 6.4


def _make_matrix_annotation(
    axis_0_label_, statistic_matrix, y, row_height, add_score_header
):

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

        score, moe, p_value, q_value = statistic_matrix[axis_0_index]

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
    series,
    dataframe,
    function_or_statistic_dataframe,
    vector_ascending=True,
    job_number=1,
    random_seed=RANDOM_SEED,
    sample_number=10,
    shuffle_number=10,
    plot=True,
    plot_number=8,
    vector_type="continuous",
    matrix_type="continuous",
    plot_standard_deviation=nan,
    title="Function Heat Map",
    directory_path=None,
):

    series = series.loc[series.index & dataframe.columns]

    if vector_ascending is not None:

        series.sort_values(ascending=vector_ascending, inplace=True)

    vector, axis_1_label_, vector_label = untangle(series)[:3]

    dataframe = dataframe.loc[:, axis_1_label_]

    matrix, axis_0_label_ = untangle(dataframe)[:2]

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

            print("0.95 Margin of Error (sample_number={})...".format(sample_number))

            row_x_sample_matrix = full((axis_0_size, sample_number), nan)

            sample_number = int(axis_1_size * SAMPLE_FRACTION)

            for sample_index in range(sample_number):

                sample_index_ = choice(axis_1_size, size=sample_number)

                vector_sample = vector[sample_index_]

                row_x_sample_matrix[:, sample_index] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    (
                        (vector_sample, row_vector_sample, function)
                        for row_vector_sample in matrix[:, sample_index_]
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

    sort_index_ = argsort(statistic_matrix[:, 0])

    statistic_matrix = statistic_matrix[sort_index_]

    matrix = matrix[sort_index_]

    axis_0_label_ = axis_0_label_[sort_index_]

    statistic_label_ = asarray(("Score", "Margin of Error", "P-Value", "Q-Value"))

    statistic_dataframe = entangle(
        statistic_matrix, axis_0_label_, statistic_label_, "Feature", "Statistic"
    )

    if directory_path is not None:

        statistic_dataframe.to_csv("{}statistic.tsv".format(directory_path), sep="\t")

    if plot:

        layout_width = 480

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
                    "height": layout_width * GOLDEN_FACTOR,
                    "width": layout_width,
                    "title": {"text": title},
                    "xaxis": {"title": {"text": "Statistic"}},
                    "yaxis": {"title": {"text": "Rank"}},
                },
            },
        )

        if plot_number is not None:

            is_extreme_ = check_is_extreme(matrix[:, 0], "<>", number=plot_number)

            statistic_matrix = statistic_matrix[is_extreme_]

            matrix = matrix[is_extreme_]

            axis_0_label_ = axis_0_label_[is_extreme_]

        if vector_type == "continuous":

            vector = normalize(vector, "-0-").clip(
                min=-plot_standard_deviation, max=plot_standard_deviation
            )

            vector_minimum = -plot_standard_deviation

            vector_maximum = plot_standard_deviation

        else:

            vector_minimum = None

            vector_maximum = None

        if matrix_type == "continuous":

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
            and check_is_not_nan(matrix).any(axis=0).all()
        ):

            leaf_index_ = []

            for number in unique(vector):

                group_index_ = where(vector == number)[0]

                leaf_index_.append(group_index_[cluster(matrix.T[group_index_])[0]])

            leaf_index_ = concatenate(leaf_index_)

            vector = vector[leaf_index_]

            matrix = matrix[:, leaf_index_]

            axis_1_label_[leaf_index_]

        row_number = matrix.shape[0] + 2

        row_height = 1 / row_number

        layout = merge(
            {
                "height": max(480, 24 * row_number),
                "title": {"text": title},
                "yaxis2": {"domain": (1 - row_height, 1), "showticklabels": False},
                "yaxis": {"domain": (0, 1 - row_height * 2), "showticklabels": False},
            },
            LAYOUT_BASE,
        )

        layout["annotations"] = _make_vector_annotation(
            vector_label, 1 - row_height / 2
        ) + _make_matrix_annotation(
            axis_0_label_, matrix, 1 - row_height / 2 * 3, row_height, True
        )

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}function_heat_map.html".format(directory_path)

        plot_plotly(
            {
                "data": [
                    {
                        "yaxis": "y2",
                        "x": axis_1_label_,
                        "z": vector.reshape((1, -1)),
                        "zmin": vector_minimum,
                        "zmax": vector_maximum,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[vector_type],
                        **HEATMAP_BASE,
                    },
                    {
                        "yaxis": "y",
                        "x": axis_1_label_,
                        "y": axis_0_label_[::-1],
                        "z": matrix[::-1],
                        "zmin": matrix_minimum,
                        "zmax": matrix_maximum,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[matrix_type],
                        **HEATMAP_BASE,
                    },
                ],
                "layout": layout,
            },
            file_path=file_path,
        )

    return statistic_dataframe


def summarize(
    series,
    data_,
    plot_only_shared=False,
    vector_ascending=True,
    vector_type="continuous",
    plot_standard_deviation=nan,
    title="Function Heat Map Summary",
    file_path=None,
):

    if plot_only_shared:

        for data in data_:

            series = series.loc[series.index & data["dataframe"].columns]

    if vector_ascending is not None:

        series.sort_values(ascending=vector_ascending, inplace=True)

    vector, axis_1_label_, vector_label = untangle(series)[:3]

    if vector_type == "continuous":

        vector = normalize(vector, "-0-").clip(
            min=-plot_standard_deviation, max=plot_standard_deviation
        )

        vector_minimum = -plot_standard_deviation

        vector_maximum = plot_standard_deviation

    else:

        vector_minimum = None

        vector_maximum = None

    row_number = 1

    space_number = 2

    for data in data_:

        row_number += data["dataframe"].shape[0] + space_number

    row_height = 1 / row_number

    layout = merge(
        {
            "height": max(480, 24 * row_number),
            "title": {"text": title},
            "annotations": _make_vector_annotation(vector_label, 1 - row_height / 2),
        },
        LAYOUT_BASE,
    )

    yaxis = "yaxis{}".format(len(data_) + 1)

    domain = 1 - row_height, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "x": axis_1_label_,
            "z": vector.reshape((1, -1)),
            "zmin": vector_minimum,
            "zmax": vector_maximum,
            "colorscale": DATA_TYPE_TO_COLORSCALE[vector_type],
            **HEATMAP_BASE,
        }
    ]

    for data_index, data in enumerate(data_):

        dataframe = data["dataframe"].reindex(labels=axis_1_label_, axis=1)

        statistic_dataframe = data["statistic_dataframe"]

        statistic_dataframe.sort_values("Score", ascending=False, inplace=True)

        dataframe = dataframe.loc[statistic_dataframe.index, :]

        matrix, axis_0_label_ = untangle(dataframe)[:2]

        #

        if data["type"] == "continuous":

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

        yaxis = "yaxis{}".format(len(data_) - data_index)

        domain = (
            max(0, domain[0] - row_height * (space_number + dataframe.shape[0])),
            domain[0] - row_height * space_number,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "x": axis_1_label_,
                "y": axis_0_label_[::-1],
                "z": matrix[::-1],
                "zmin": matrix_minimum,
                "zmax": matrix_maximum,
                "colorscale": DATA_TYPE_TO_COLORSCALE[data["type"]],
                **HEATMAP_BASE,
            }
        )

        y = domain[1] + row_height / 2

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": y,
                "xanchor": "center",
                "text": "<b>{}</b>".format(data["name"]),
                **ANNOTATION_BASE,
            }
        )

        layout["annotations"] += _make_matrix_annotation(
            axis_0_label_, statistic_matrix, y, row_height, data_index == 0
        )

    plot_plotly({"data": data, "layout": layout}, file_path=file_path)
