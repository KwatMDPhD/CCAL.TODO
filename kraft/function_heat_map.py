from multiprocessing import Pool

from numpy import asarray, full, nan, unique, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame

from .array import (
    check_is_extreme,
    function_on_1_number_array_not_nan,
    function_on_2_number_array_not_nan,
    normalize,
)
from .clustering import cluster
from .CONSTANT import RANDOM_SEED, SAMPLE_FRACTION
from .dictionary import merge
from .plot import DATA_TYPE_TO_COLORSCALE, plot_plotly
from .significance import get_margin_of_error, get_p_value_and_q_value

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


def _process_vector(vector, data_type, plot_standard_deviation):

    vector = vector.copy()

    minimum = maximum = None

    if data_type == "continuous":

        vector = normalize(vector, "-0-").clip(
            min=-plot_standard_deviation, max=plot_standard_deviation
        )

        minimum = -plot_standard_deviation

        maximum = plot_standard_deviation

    return vector, minimum, maximum


def _process_matrix(matrix, data_type, plot_standard_deviation):

    matrix = matrix.copy()

    minimum = maximum = None

    if data_type == "continuous":

        for index in range(matrix.shape[0]):

            matrix[index] = normalize(matrix[index], "-0-").clip(
                min=-plot_standard_deviation, max=plot_standard_deviation
            )

        minimum = -plot_standard_deviation

        maximum = plot_standard_deviation

    return matrix, minimum, maximum


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
    # F
    dataframe,
    function_or_statistic_dataframe,
    vector_ascending=True,
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

    # A
    series = series.loc[series.index & dataframe.columns]

    # B
    if vector_ascending is not None:

        series.sort_values(ascending=vector_ascending, inplace=True)

    # C
    vector = series.to_numpy()

    # D
    vector_name = series.name

    # E
    axis_1_label_ = series.index.to_numpy()

    # G
    dataframe.reindex(labels=axis_1_label_, axis=1, inplace=True)

    if callable(function_or_statistic_dataframe):

        matrix = dataframe.to_numpy()

        function = function_or_statistic_dataframe

        axis_0_size, axis_1_size = matrix.shape

        pool = Pool(job_number)

        seed(seed=random_seed)

        print("Score (function={})...".format(function.__name__))

        score_vector = asarray(
            pool.starmap(
                function_on_2_number_array_not_nan,
                ((vector, row, function) for row in matrix),
            )
        )

        if 0 < sample_number:

            print("0.95 Margin of Error (sample_number={})...".format(sample_number))

            _x_sample_matrix = full((axis_0_size, sample_number), nan)

            sample_number = int(axis_1_size * SAMPLE_FRACTION)

            for sample_index in range(sample_number):

                axis_1_index_ = choice(axis_1_size, size=sample_number)

                vector_sample = vector[axis_1_index_]

                _x_sample_matrix[:, sample_index] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    (
                        (vector_sample, row, function)
                        for row in matrix[:, axis_1_index_]
                    ),
                )

            margin_of_error_vector = asarray(
                tuple(
                    function_on_1_number_array_not_nan(row, get_margin_of_error)
                    for row in _x_sample_matrix
                )
            )

        if 0 < shuffle_number:

            print("P-Value and Q-Value (shuffle_number={})...".format(shuffle_number))

            _x_shuffle_matrix = full((axis_0_size, shuffle_number), nan)

            vector_shuffle = vector.copy()

            for shuffle_index in range(shuffle_number):

                shuffle(vector_shuffle)

                _x_shuffle_matrix[:, shuffle_index] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    ((vector_shuffle, row, function) for row in matrix),
                )

            p_value_vector, q_value_vector = get_p_value_and_q_value(
                score_vector, _x_shuffle_matrix.ravel(), "<>"
            )

        pool.terminate()

        statistic_dataframe = DataFrame(
            data=asarray(
                (score_vector, margin_of_error_vector, p_value_vector, q_value_vector)
            ).T,
            index=dataframe.index,
            columns=("Score", "Margin of Error", "P-Value", "Q-Value"),
        )

    else:

        # H
        statistic_dataframe = function_or_statistic_dataframe.loc[dataframe.index, :]

    # I
    statistic_dataframe.sort_values("Score", ascending=False, inplace=True)

    if directory_path is not None:

        statistic_dataframe.to_csv("{}statistic.tsv".format(directory_path), sep="\t")

    # J
    dataframe = dataframe.loc[statistic_dataframe.index, :]

    if plot:

        # K
        matrix = dataframe.to_numpy()

        # L
        axis_0_label_ = dataframe.index.to_numpy()

        # M
        statistic_matrix = statistic_dataframe.to_numpy()

        if plot_number is not None:

            is_extreme_ = check_is_extreme(
                statistic_matrix[:, 0], "<>", number=plot_number
            )

            statistic_matrix = statistic_matrix[is_extreme_]

            matrix = matrix[is_extreme_]

            axis_0_label_ = axis_0_label_[is_extreme_]

        # X
        vector, vector_minimum, vector_maximum = _process_vector(
            vector, vector_data_type, plot_standard_deviation
        )

        # Y
        matrix, matrix_minimum, matrix_maximum = _process_matrix(
            matrix, matrix_data_type, plot_standard_deviation
        )

        for number, count in unique(vector, return_counts=True):

            if 2 < count:

                group_index_ = where(vector == number)[0]

                leaf_index_ = cluster(matrix.T[group_index_])[0]

                group_index_new_ = group_index_[leaf_index_]

                axis_1_label_[group_index_] = axis_1_label_[group_index_new_]

                vector[group_index_] = vector[group_index_new_]

                matrix[:, group_index_] = matrix[:, group_index_new_]

        row_number = matrix.shape[0] + 2

        row_height = 1 / row_number

        # P
        layout = merge(
            {
                "height": max(480, 24 * row_number),
                "title": {"text": title},
                "yaxis2": {"domain": (1 - row_height, 1), "showticklabels": False},
                "yaxis": {"domain": (0, 1 - row_height * 2), "showticklabels": False},
                "annotations": _make_vector_annotation(vector_name, 1 - row_height / 2),
            },
            LAYOUT_BASE,
        )

        # Q
        layout["annotations"] += _make_matrix_annotation(
            axis_0_label_, statistic_matrix, 1 - row_height / 2 * 3, row_height, True
        )

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}function_heat_map.html".format(directory_path)

        plot_plotly(
            {
                "data": [
                    # N
                    {
                        "yaxis": "y2",
                        "z": vector.reshape((1, -1)),
                        "x": axis_1_label_,
                        "zmin": vector_minimum,
                        "zmax": vector_maximum,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[vector_data_type],
                        **HEATMAP_BASE,
                    },
                    # O
                    {
                        "yaxis": "y",
                        "z": matrix[::-1],
                        "x": axis_1_label_,
                        "y": axis_0_label_[::-1],
                        "zmin": matrix_minimum,
                        "zmax": matrix_maximum,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[matrix_data_type],
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
    intersect=True,
    vector_ascending=True,
    vector_data_type="continuous",
    plot_standard_deviation=nan,
    title="Function Heat Map Summary",
    file_path=None,
):

    # A
    if intersect:

        for data in data_:

            series = series.loc[series.index & data["dataframe"].columns]

    # B
    if vector_ascending is not None:

        series.sort_values(ascending=vector_ascending, inplace=True)

    # C
    vector = series.to_numpy()

    # D
    vector_name = series.name

    # E
    axis_1_label_ = series.index.to_numpy()

    # X
    vector, vector_minimum, vector_maximum = _process_vector(
        vector, vector_data_type, plot_standard_deviation
    )

    row_number = 1

    space_number = 2

    for data in data_:

        row_number += data["dataframe"].shape[0] + space_number

    row_height = 1 / row_number

    # P
    layout = merge(
        {
            "height": max(480, 24 * row_number),
            "title": {"text": title},
            "annotations": _make_vector_annotation(vector_name, 1 - row_height / 2),
        },
        LAYOUT_BASE,
    )

    yaxis = "yaxis{}".format(len(data_) + 1)

    domain = 1 - row_height, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    data = [
        # N
        {
            "yaxis": yaxis.replace("axis", ""),
            "z": vector.reshape((1, -1)),
            "x": axis_1_label_,
            "zmin": vector_minimum,
            "zmax": vector_maximum,
            "colorscale": DATA_TYPE_TO_COLORSCALE[vector_data_type],
            **HEATMAP_BASE,
        }
    ]

    for data_index, data in enumerate(data_):

        # F
        dataframe = data["dataframe"]

        # G
        dataframe.reindex(labels=axis_1_label_, axis=1, inplace=True)

        # H
        statistic_dataframe = data["statistic_dataframe"].loc[dataframe.index, :]

        # I
        statistic_dataframe.sort_values("Score", ascending=False, inplace=True)

        # J
        dataframe = dataframe.loc[statistic_dataframe.index, :]

        # K
        matrix = dataframe.to_numpy()

        # L
        axis_0_label_ = dataframe.index.to_numpy()

        # M
        statistic_matrix = statistic_dataframe.to_numpy()

        # Y
        matrix, matrix_minimum, matrix_maximum = _process_matrix(
            matrix, data["type"], plot_standard_deviation
        )

        yaxis = "yaxis{}".format(len(data_) - data_index)

        domain = (
            max(0, domain[0] - row_height * (space_number + dataframe.shape[0])),
            domain[0] - row_height * space_number,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            # O
            {
                "yaxis": yaxis.replace("axis", ""),
                "z": matrix[::-1],
                "x": axis_1_label_,
                "y": axis_0_label_[::-1],
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

        # Q
        layout["annotations"] += _make_matrix_annotation(
            axis_0_label_, statistic_matrix, y, row_height, data_index == 0
        )

    plot_plotly({"data": data, "layout": layout}, file_path=file_path)
