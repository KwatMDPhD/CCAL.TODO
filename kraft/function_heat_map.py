from math import ceil
from multiprocessing import Pool

from numpy import apply_along_axis, asarray, concatenate, full, isnan, nan, where
from numpy.random import choice, seed, shuffle
from pandas import DataFrame, Series, unique

from .array import ignore_nan_and_function_1, ignore_nan_and_function_2, normalize
from .clustering import cluster
from .CONSTANT import RANDOM_SEED
from .dict_ import merge
from .plot import DATA_TYPE_TO_COLORSCALE, plot_plotly
from .series import get_extreme_labels
from .significance import get_moe, get_p_values_and_q_values


def _get_x(score_index):

    return 1.1 + score_index / 6.4


def function_heat_map(
    series,
    dataframe,
    function,
    series_ascending=True,
    scores=None,
    n_job=1,
    random_seed=RANDOM_SEED,
    n_sampling=10,
    n_permutation=10,
    score_ascending=False,
    plot=True,
    n_extreme=8,
    fraction_extreme=None,
    series_data_type="continuous",
    dataframe_data_type="continuous",
    plot_std=nan,
    layout=None,
    directory_path=None,
):

    common_pandas_index = series.index & dataframe.columns

    print(
        "series.index ({}) and dataframe.columns ({}) have {} in common.".format(
            series.index.size, dataframe.columns.size, common_pandas_index.size
        )
    )

    series = series[common_pandas_index]

    if series_ascending is not None:

        series.sort_values(ascending=series_ascending, inplace=True)

    dataframe = dataframe[series.index]

    if scores is None:

        scores = DataFrame(
            index=dataframe.index, columns=("Score", "0.95 MoE", "P-Value", "Q-Value")
        )

        n_row, n_column = dataframe.shape

        n_job = min(n_job, n_row)

        print("Computing scores (n_job={})...".format(n_job))

        pool = Pool(n_job)

        print("\tScore (function={})...".format(function.__name__))

        seed(seed=random_seed)

        vector = series.values

        scores["Score"] = asarray(
            pool.starmap(
                ignore_nan_and_function_2,
                ((vector, matrix_row, function) for matrix_row in dataframe.values),
            )
        )

        if 0 < n_sampling:

            print("\t0.95 MoE (n_sampling={})...".format(n_sampling))

            row_x_sampling = full((n_row, n_sampling), nan)

            n_column_to_sample = ceil(n_column * 0.632)

            for sampling_index in range(n_sampling):

                row_name = choice(n_column, size=n_column_to_sample)

                # TODO: confirm that initializing outside tuple comprehension is faster
                vector = series.values[row_name]

                row_x_sampling[:, sampling_index] = pool.starmap(
                    ignore_nan_and_function_2,
                    (
                        (vector, matrix_row, function)
                        for matrix_row in dataframe.values[:, row_name]
                    ),
                )

            scores["0.95 MoE"] = apply_along_axis(
                lambda scores: get_moe(scores[~isnan(scores)]), 1, row_x_sampling,
            )

        if 0 < n_permutation:

            print("\tP-Value and Q-Value (n_permutation={})...".format(n_permutation))

            row_x_permutation = full((n_row, n_permutation), nan)

            vector = series.values.copy()

            for permuting_index in range(n_permutation):

                shuffle(vector)

                row_x_permutation[:, permuting_index] = pool.starmap(
                    ignore_nan_and_function_2,
                    ((vector, matrix_row, function) for matrix_row in dataframe.values),
                )

            scores["P-Value"], scores["Q-Value"] = get_p_values_and_q_values(
                scores["Score"].values, row_x_permutation.flatten(), "<>"
            )

        pool.terminate()

    else:

        scores = scores.reindex(row_name=dataframe.index)

    scores.sort_values("Score", ascending=score_ascending, inplace=True)

    if directory_path is not None:

        tsv_file_path = "{}/scores.tsv".format(directory_path)

        scores.to_csv(tsv_file_path, sep="\t")

    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Scores"},
                    "xaxis": {"title": {"text": "Rank"}},
                },
                "data": [
                    {
                        "name": score_name,
                        "x": score_values.index,
                        "y": score_values.values,
                    }
                    for score_name, score_values in scores.items()
                ],
            },
        )

        series_plot = series.copy()

        scores_plot = scores.copy()

        if n_extreme is not None or fraction_extreme is not None:

            scores_plot = scores_plot.loc[
                get_extreme_labels(
                    scores_plot["Score"],
                    "<>",
                    n=n_extreme,
                    fraction=fraction_extreme,
                    plot=False,
                )
            ].sort_values("Score", ascending=score_ascending)

        dataframe_plot = dataframe.loc[scores_plot.index]

        if series_data_type == "continuous":

            series_plot = Series(
                ignore_nan_and_function_1(
                    series_plot.values, normalize, "-0-", update=True
                ),
                name=series_plot.name,
                index=series_plot.index,
            ).clip(lower=-plot_std, upper=plot_std)

        if dataframe_data_type == "continuous":

            dataframe_plot = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1,
                    1,
                    dataframe_plot.values,
                    normalize,
                    "-0-",
                    update=True,
                ),
                index=dataframe_plot.index,
                columns=dataframe_plot.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        if (
            not series_plot.isna().any()
            and isinstance(series_ascending, bool)
            and 1 < series_plot.value_counts().min()
        ):

            print("Clustering within category...")

            vector = series_plot.values

            matrix = dataframe_plot.values

            leave_index = []

            for n in unique(vector):

                row_name = where(vector == n)[0]

                leave_index.append(row_name[cluster(matrix.T[row_name])[0]])

            dataframe_plot = dataframe_plot.iloc[:, concatenate(leave_index)]

            series_plot = series_plot[dataframe_plot.columns]

        n_row_plot = 1 + 1 + dataframe_plot.shape[0]

        fraction_row = 1 / n_row_plot

        layout_template = {
            "height": max(480, 24 * n_row_plot),
            "width": 800,
            "margin": {"l": 200, "r": 200},
            "title": {"x": 0.5},
            "yaxis": {"domain": (0, 1 - 2 * fraction_row), "showticklabels": False},
            "yaxis2": {"domain": (1 - fraction_row, 1), "showticklabels": False},
            "annotations": [],
        }

        if layout is not None:

            layout = merge(layout_template, layout)

        annotation_template = {
            "xref": "paper",
            "yref": "paper",
            "yanchor": "middle",
            "font": {"size": 10},
            "showarrow": False,
        }

        y = 1 - fraction_row / 2

        layout["annotations"].append(
            {
                "x": 0,
                "y": y,
                "xanchor": "right",
                "text": "<b>{}</b>".format(series_plot.name),
                **annotation_template,
            }
        )

        y -= fraction_row

        for score_index, score_name in enumerate(
            ("Score (\u0394)", "P-Value", "Q-Value")
        ):

            layout["annotations"].append(
                {
                    "x": _get_x(score_index),
                    "y": y,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(score_name),
                    **annotation_template,
                }
            )

        y -= fraction_row

        for row_name, (score, moe, p_value, q_value) in scores_plot.iterrows():

            layout["annotations"].append(
                {
                    "x": 0,
                    "y": y,
                    "xanchor": "right",
                    "text": row_name,
                    **annotation_template,
                }
            )

            for score_index, score_str in enumerate(
                (
                    "{:.2f} ({:.2f})".format(score, moe),
                    "{:.2e}".format(p_value),
                    "{:.2e}".format(q_value),
                )
            ):

                layout["annotations"].append(
                    {
                        "x": _get_x(score_index),
                        "y": y,
                        "xanchor": "center",
                        "text": score_str,
                        **annotation_template,
                    }
                )

            y -= fraction_row

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = tsv_file_path.replace(".tsv", ".html")

        heatmap_trace_template = {
            "type": "heatmap",
            "zmin": -plot_std,
            "zmax": plot_std,
            "showscale": False,
        }

        plot_plotly(
            {
                "layout": layout,
                "data": [
                    {
                        "yaxis": "y2",
                        "x": series_plot.index,
                        "z": series_plot.to_frame().T,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[series_data_type],
                        **heatmap_trace_template,
                    },
                    {
                        "yaxis": "y",
                        "x": dataframe_plot.columns,
                        "y": dataframe_plot.index[::-1],
                        "z": dataframe_plot.iloc[::-1],
                        "colorscale": DATA_TYPE_TO_COLORSCALE[dataframe_data_type],
                        **heatmap_trace_template,
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return scores


def function_heat_map_summary(
    series,
    dataframe_dicts,
    scores,
    plot_only_shared=False,
    series_ascending=True,
    series_data_type="continuous",
    plot_std=nan,
    html_file_path=None,
):

    if plot_only_shared:

        for dataframe_dict in dataframe_dicts.values():

            series = series.loc[series.index & dataframe_dict["dataframe"].columns]

    if series_ascending is not None:

        series.sort_values(ascending=series_ascending, inplace=True)

    series_plot = series.copy()

    if series_data_type == "continuous":

        series_plot = Series(
            ignore_nan_and_function_1(
                series_plot.values, normalize, "-0-", update=True
            ),
            name=series_plot.name,
            index=series_plot.index,
        ).clip(lower=-plot_std, upper=plot_std)

    n_space = 2

    n_row_plot = 1

    for dataframe_dict in dataframe_dicts.values():

        n_row_plot += n_space

        n_row_plot += dataframe_dict["dataframe"].shape[0]

    layout = {
        "height": max(480, 24 * n_row_plot),
        "width": 800,
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        "annotations": [],
    }

    fraction_row = 1 / n_row_plot

    yaxis = "yaxis{}".format(len(dataframe_dicts) + 1)

    domain = 1 - fraction_row, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    heatmap_trace_template = {
        "type": "heatmap",
        "zmin": -plot_std,
        "zmax": plot_std,
        "showscale": False,
    }

    data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "x": series_plot.index,
            "z": series_plot.to_frame().T,
            "colorscale": DATA_TYPE_TO_COLORSCALE[series_data_type],
            **heatmap_trace_template,
        }
    ]

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
            "y": 1 - fraction_row / 2,
            "xanchor": "right",
            "text": "<b>{}</b>".format(series.name),
            **annotation_template,
        }
    )

    for dataframe_index, (dataframe_name, dataframe_dict) in enumerate(
        dataframe_dicts.items()
    ):

        dataframe_plot = dataframe_dict["dataframe"].reindex(columns=series_plot.index)

        scores_plot = scores[dataframe_name].reindex(index=dataframe_plot.index)

        if "emphasis" in dataframe_dict:

            score_ascending = dataframe_dict["emphasis"] == "-"

        else:

            score_ascending = False

        scores_plot.sort_values("Score", ascending=score_ascending, inplace=True)

        dataframe_plot = dataframe_plot.loc[scores_plot.index]

        if dataframe_dict["data_type"] == "continuous":

            dataframe_plot = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1,
                    1,
                    dataframe_plot.values,
                    normalize,
                    "-0-",
                    update=True,
                ),
                index=dataframe_plot.index,
                columns=dataframe_plot.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        yaxis = "yaxis{}".format(len(dataframe_dicts) - dataframe_index)

        domain = (
            max(0, domain[0] - fraction_row * (n_space + dataframe_plot.shape[0])),
            domain[0] - fraction_row * n_space,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "x": dataframe_plot.columns,
                "y": dataframe_plot.index[::-1],
                "z": dataframe_plot.values[::-1],
                "colorscale": DATA_TYPE_TO_COLORSCALE[dataframe_dict["data_type"]],
                **heatmap_trace_template,
            }
        )

        y = domain[1] + fraction_row / 2

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": y,
                "xanchor": "center",
                "text": "<b>{}</b>".format(dataframe_name),
                **annotation_template,
            }
        )

        if dataframe_index == 0:

            for score_index, score_str in enumerate(
                ("Score (\u0394)", "P-Value", "Q-Value")
            ):

                layout["annotations"].append(
                    {
                        "x": _get_x(score_index),
                        "y": y,
                        "xanchor": "center",
                        "text": "<b>{}</b>".format(score_str),
                        **annotation_template,
                    }
                )

        y -= fraction_row

        for row_name, (score, moe, p_value, q_value) in scores_plot.iterrows():

            layout["annotations"].append(
                {
                    "x": 0,
                    "y": y,
                    "xanchor": "right",
                    "text": row_name,
                    **annotation_template,
                }
            )

            for score_index, score_str in enumerate(
                (
                    "{:.2f} ({:.2f})".format(score, moe),
                    "{:.2e}".format(p_value),
                    "{:.2e}".format(q_value),
                )
            ):

                layout["annotations"].append(
                    {
                        "x": _get_x(score_index),
                        "y": y,
                        "xanchor": "center",
                        "text": score_str,
                        **annotation_template,
                    }
                )

            y -= fraction_row

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
