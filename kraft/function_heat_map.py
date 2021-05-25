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
from .dict_a import merge
from .plot import (
    BINARY_COLORSCALE,
    CATEGORICAL_COLORSCALE,
    CONTINUOUS_COLORSCALE,
    plot_plotly,
)
from .significance import get_moe, get_p_value_and_q_value

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

DATA_TYPE_TO_COLORSCALE = {
    "continuous": CONTINUOUS_COLORSCALE,
    "categorical": CATEGORICAL_COLORSCALE,
    "binary": BINARY_COLORSCALE,
}


def _process_vector_for_plot(vector, data_type, plot_std):

    if data_type == "continuous":

        if 0 < vector.std():

            vector = function_on_1_number_array_not_nan(
                vector, normalize, "-0-", update=True
            ).clip(min=-plot_std, max=plot_std)

        return vector, -plot_std, plot_std

    return vector.copy(), None, None


def _process_matrix_for_plot(matrix, data_type, plot_std):

    matrix = matrix.copy()

    if data_type == "continuous":

        for index in range(matrix.shape[0]):

            matrix[index] = _process_vector_for_plot(
                matrix[index], data_type, plot_std
            )[0]

        return matrix, -plot_std, plot_std

    return matrix, None, None


def _make_vector_annotation(vector_name, y):

    return [
        {
            "y": y,
            "x": 0,
            "xanchor": "right",
            "text": "<b>{}</b>".format(vector_name),
            **ANNOTATION_BASE,
        }
    ]


def _get_statistic_x(index):

    return 1.08 + index / 6.4


def _make_matrix_annotation(
    axis_0_label_, statistic_matrix, y, row_height, add_score_header
):

    annotation_ = []

    if add_score_header:

        for index, statistic in enumerate(("Score (\u0394)", "P-Value", "Q-Value")):

            annotation_.append(
                {
                    "y": y,
                    "x": _get_statistic_x(index),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(statistic),
                    **ANNOTATION_BASE,
                }
            )

    y -= row_height

    for index in range(axis_0_label_.size):

        annotation_.append(
            {
                "y": y,
                "x": 0,
                "xanchor": "right",
                "text": axis_0_label_[index],
                **ANNOTATION_BASE,
            }
        )

        score, moe, p_value, q_value = statistic_matrix[index]

        for index, statistic in enumerate(
            (
                "{:.2f} ({:.2f})".format(score, moe),
                "{:.2e}".format(p_value),
                "{:.2e}".format(q_value),
            )
        ):

            annotation_.append(
                {
                    "y": y,
                    "x": _get_statistic_x(index),
                    "xanchor": "center",
                    "text": statistic,
                    **ANNOTATION_BASE,
                }
            )

        y -= row_height

    return annotation_


def make(
    se,
    df,
    function_or_statistic_df,
    vector_ascending=True,
    n_job=1,
    random_seed=RANDOM_SEED,
    n_sample=10,
    n_shuffle=10,
    plot=True,
    n_plot=8,
    vector_data_type="continuous",
    matrix_data_type="continuous",
    plot_std=nan,
    title="Function Heat Map",
    directory_path=None,
):

    #
    se = se.loc[se.index.intersection(df.columns)]

    #
    if vector_ascending is not None:

        se.sort_values(ascending=vector_ascending, inplace=True)

    #
    vector = se.to_numpy()

    #
    vector_name = se.name

    #
    axis_1_label_ = se.index.to_numpy()

    #
    df = df.reindex(labels=axis_1_label_, axis=1)

    if callable(function_or_statistic_df):

        matrix = df.to_numpy()

        matrix_axis_0_size, axis_1_size = matrix.shape

        function = function_or_statistic_df

        pool = Pool(n_job)

        seed(seed=random_seed)

        print("Score ({})...".format(function.__name__))

        score_ = asarray(
            pool.starmap(
                function_on_2_number_array_not_nan,
                ((vector, row, function) for row in matrix),
            )
        )

        if 0 < n_sample:

            print("0.95 MoE ({} sample)...".format(n_sample))

            row_x_sample = full((matrix_axis_0_size, n_sample), nan)

            choice_n = int(axis_1_size * SAMPLE_FRACTION)

            for sample_index in range(n_sample):

                sample_index_ = choice(axis_1_size, size=choice_n, replace=False)

                vector_sample = vector[sample_index_]

                row_x_sample[:, sample_index] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    (
                        (vector_sample, row, function)
                        for row in matrix[:, sample_index_]
                    ),
                )

            moe_ = asarray(
                tuple(
                    function_on_1_number_array_not_nan(row, get_moe)
                    for row in row_x_sample
                )
            )

        else:

            moe_ = full(score_.size, nan)

        if 0 < n_shuffle:

            print("P-Value and Q-Value ({} shuffle)...".format(n_shuffle))

            row_x_shuffle = full((matrix_axis_0_size, n_shuffle), nan)

            vector_shuffle = vector.copy()

            for shuffle_index in range(n_shuffle):

                shuffle(vector_shuffle)

                row_x_shuffle[:, shuffle_index] = pool.starmap(
                    function_on_2_number_array_not_nan,
                    ((vector_shuffle, row, function) for row in matrix),
                )

            p_value_, q_value_ = get_p_value_and_q_value(
                score_, row_x_shuffle.ravel(), "<>"
            )

        else:

            p_value_ = q_value_ = full(score_.size, nan)

        pool.terminate()

        statistic_df = DataFrame(
            data=asarray((score_, moe_, p_value_, q_value_)).T,
            index=df.index,
            columns=("Score", "MoE", "P-Value", "Q-Value"),
        )

    else:

        #
        statistic_df = function_or_statistic_df.loc[df.index, :]

    #
    statistic_df.sort_values("Score", ascending=False, inplace=True)

    if directory_path is not None:

        statistic_df.to_csv("{}statistic.tsv".format(directory_path), sep="\t")

    #
    df = df.loc[statistic_df.index, :]

    if plot:

        #
        matrix = df.to_numpy()

        #
        axis_0_label_ = df.index.to_numpy()

        #
        statistic_matrix = statistic_df.to_numpy()

        if n_plot is not None and n_plot < statistic_matrix.shape[0]:

            is_ = check_is_extreme(statistic_matrix[:, 0], "<>", n=n_plot)

            statistic_matrix = statistic_matrix[is_]

            matrix = matrix[is_]

            axis_0_label_ = axis_0_label_[is_]

        #
        vector, vector_min, vector_max = _process_vector_for_plot(
            vector, vector_data_type, plot_std
        )

        #
        matrix, matrix_min, matrix_max = _process_matrix_for_plot(
            matrix, matrix_data_type, plot_std
        )

        if vector_data_type != "continuous":

            for number, count in zip(*unique(vector, return_counts=True)):

                if 2 < count:

                    print("Clustering group {}...".format(number))

                    index_ = where(vector == number)[0]

                    index_cluster_ = index_[cluster(matrix.T[index_])[0]]

                    vector[index_] = vector[index_cluster_]

                    matrix[:, index_] = matrix[:, index_cluster_]

                    axis_1_label_[index_] = axis_1_label_[index_cluster_]

        row_n = matrix.shape[0] + 2

        row_height = 1 / row_n

        #
        layout = merge(
            {
                "height": max(480, 24 * row_n),
                "title": {"text": title},
                "yaxis2": {"domain": (1 - row_height, 1), "showticklabels": False},
                "yaxis": {"domain": (0, 1 - row_height * 2), "showticklabels": False},
                "annotations": _make_vector_annotation(vector_name, 1 - row_height / 2),
            },
            LAYOUT_BASE,
        )

        #
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
                    #
                    {
                        "yaxis": "y2",
                        "z": vector.reshape((1, -1)),
                        "x": axis_1_label_,
                        "zmin": vector_min,
                        "zmax": vector_max,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[vector_data_type],
                        **HEATMAP_BASE,
                    },
                    #
                    {
                        "yaxis": "y",
                        "z": matrix[::-1],
                        "y": axis_0_label_[::-1],
                        "x": axis_1_label_,
                        "zmin": matrix_min,
                        "zmax": matrix_max,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[matrix_data_type],
                        **HEATMAP_BASE,
                    },
                ],
                "layout": layout,
            },
            file_path=file_path,
        )

    return statistic_df


def summarize(
    se,
    data_,
    intersect=True,
    vector_ascending=True,
    vector_data_type="continuous",
    plot_std=nan,
    title="Function Heat Map Summary",
    file_path=None,
):

    #
    if intersect:

        for data in data_:

            se = se.loc[se.index.intersection(data["df"].columns)]

    #
    if vector_ascending is not None:

        se.sort_values(ascending=vector_ascending, inplace=True)

    #
    vector = se.to_numpy()

    #
    vector_name = se.name

    #
    axis_1_label_ = se.index.to_numpy()

    #
    vector, vector_min, vector_max = _process_vector_for_plot(
        vector, vector_data_type, plot_std
    )

    row_n = 1

    space_n = 2

    for data in data_:

        row_n += data["df"].shape[0] + space_n

    row_height = 1 / row_n

    #
    layout = merge(
        {
            "height": max(480, 24 * row_n),
            "title": {"text": title},
            "annotations": _make_vector_annotation(vector_name, 1 - row_height / 2),
        },
        LAYOUT_BASE,
    )

    data_n = len(data_)

    yaxis = "yaxis{}".format(data_n + 1)

    domain = 1 - row_height, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    data = [
        #
        {
            "yaxis": yaxis.replace("axis", ""),
            "z": vector.reshape((1, -1)),
            "x": axis_1_label_,
            "zmin": vector_min,
            "zmax": vector_max,
            "colorscale": DATA_TYPE_TO_COLORSCALE[vector_data_type],
            **HEATMAP_BASE,
        }
    ]

    for index, b in enumerate(data_):

        #
        df = b["df"]

        #
        df = df.reindex(labels=axis_1_label_, axis=1)

        #
        statistic_df = b["statistic_df"].loc[df.index, :]

        #
        statistic_df.sort_values("Score", ascending=False, inplace=True)

        #
        df = df.loc[statistic_df.index, :]

        #
        matrix = df.to_numpy()

        #
        axis_0_label_ = df.index.to_numpy()

        #
        statistic_matrix = statistic_df.to_numpy()

        #
        matrix, matrix_min, matrix_max = _process_matrix_for_plot(
            matrix, b["data_type"], plot_std
        )

        yaxis = "yaxis{}".format(data_n - index)

        domain = (
            max(0, domain[0] - row_height * (space_n + df.shape[0])),
            domain[0] - row_height * space_n,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            #
            {
                "yaxis": yaxis.replace("axis", ""),
                "z": matrix[::-1],
                "y": axis_0_label_[::-1],
                "x": axis_1_label_,
                "zmin": matrix_min,
                "zmax": matrix_max,
                "colorscale": DATA_TYPE_TO_COLORSCALE[b["data_type"]],
                **HEATMAP_BASE,
            }
        )

        y = domain[1] + row_height / 2

        layout["annotations"].append(
            {
                "y": y,
                "x": 0.5,
                "xanchor": "center",
                "text": "<b>{}</b>".format(b["name"]),
                **ANNOTATION_BASE,
            }
        )

        #
        layout["annotations"] += _make_matrix_annotation(
            axis_0_label_, statistic_matrix, y, row_height, index == 0
        )

    plot_plotly({"data": data, "layout": layout}, file_path=file_path)
