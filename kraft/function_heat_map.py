from math import ceil
from multiprocessing import Pool

from numpy import (
    apply_along_axis,
    asarray,
    concatenate,
    full,
    isnan,
    nan,
    unique,
    where,
)
from numpy.random import choice, seed, shuffle
from pandas import DataFrame, Series

from .array import (
    check_is_sorted,
    ignore_nan_and_function_1,
    ignore_nan_and_function_2,
    normalize,
)
from .clustering import cluster
from .CONSTANT import RANDOM_SEED
from .dict_ import merge
from .plot import DATA_TYPE_TO_COLORSCALE, plot_plotly
from .series import get_extreme_labels
from .significance import get_moe, get_p_values_and_q_values


def get_x(score_index):

    return 1.1 + score_index / 6.4


def make(
    se,
    df,
    function,
    se_ascending=True,
    scores=None,
    n_job=1,
    random_seed=RANDOM_SEED,
    n_sampling=10,
    n_permutation=10,
    score_ascending=False,
    plot=True,
    n_extreme=8,
    se_data_type="continuous",
    df_data_type="continuous",
    plot_std=nan,
    layout=None,
    directory_path=None,
):

    common_labels = se.index & df.columns

    print(
        "se.index ({}) and df.columns ({}) have {} in common.".format(
            se.index.size, df.columns.size, common_labels.size
        )
    )

    se = se.loc[common_labels]

    if se_ascending is not None:

        se.sort_values(ascending=se_ascending, inplace=True)

    df = df.loc[:, se.index]

    if scores is None:

        scores = DataFrame(
            index=df.index, columns=("Score", "0.95 MoE", "P-Value", "Q-Value")
        )

        n_row, n_column = df.shape

        n_job = min(n_row, n_job)

        print("Computing scores (n_job={})...".format(n_job))

        pool = Pool(n_job)

        print("\tScore (function={})...".format(function.__name__))

        seed(seed=random_seed)

        vector = se.to_numpy()

        scores["Score"] = asarray(
            pool.starmap(
                ignore_nan_and_function_2,
                ((vector, matrix_row, function) for matrix_row in df.to_numpy()),
            )
        )

        if 0 < n_sampling:

            print("\t0.95 MoE (n_sampling={})...".format(n_sampling))

            row_x_sampling = full((n_row, n_sampling), nan)

            n_column_to_sample = ceil(n_column * 0.632)

            for i in range(n_sampling):

                is_ = choice(n_column, size=n_column_to_sample)

                vector = se.to_numpy()[is_]

                row_x_sampling[:, i] = pool.starmap(
                    ignore_nan_and_function_2,
                    (
                        (vector, matrix_row, function)
                        for matrix_row in df.to_numpy()[:, is_]
                    ),
                )

            scores["0.95 MoE"] = apply_along_axis(
                lambda scores: get_moe(scores[~isnan(scores)]), 1, row_x_sampling,
            )

        if 0 < n_permutation:

            print("\tP-Value and Q-Value (n_permutation={})...".format(n_permutation))

            row_x_permutation = full((n_row, n_permutation), nan)

            vector = se.to_numpy().copy()

            for i in range(n_permutation):

                shuffle(vector)

                row_x_permutation[:, i] = pool.starmap(
                    ignore_nan_and_function_2,
                    ((vector, matrix_row, function) for matrix_row in df.to_numpy()),
                )

            scores["P-Value"], scores["Q-Value"] = get_p_values_and_q_values(
                scores["Score"].to_numpy(), row_x_permutation.flatten(), "<>"
            )

        pool.terminate()

    else:

        scores = scores.reindex(index=df.index)

    scores.sort_values("Score", ascending=score_ascending, inplace=True)

    if directory_path is not None:

        file_path = "{}/scores.tsv".format(directory_path)

        scores.to_csv(file_path, sep="\t")

    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": "Scores"},
                    "xaxis": {"title": {"text": "Rank"}},
                },
                "data": [
                    {
                        "name": name,
                        "x": numbers.index.to_numpy(),
                        "y": numbers.to_numpy(),
                    }
                    for name, numbers in scores.items()
                ],
            },
        )

        se_plot = se.copy()

        scores_plot = scores.copy()

        if n_extreme is not None:

            scores_plot = scores_plot.loc[
                get_extreme_labels(scores_plot["Score"], "<>", n=n_extreme, plot=False)
            ].sort_values("Score", ascending=score_ascending)

        df_plot = df.loc[scores_plot.index, :]

        if se_data_type == "continuous":

            se_plot = Series(
                ignore_nan_and_function_1(
                    se_plot.to_numpy(), normalize, "-0-", update=True
                ).clip(min=-plot_std, max=plot_std),
                index=se_plot.index,
                name=se_plot.name,
            )

        if df_data_type == "continuous":

            df_plot = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1,
                    1,
                    df_plot.to_numpy(),
                    normalize,
                    "-0-",
                    update=True,
                ).clip(min=-plot_std, max=plot_std),
                index=df_plot.index,
                columns=df_plot.columns,
            )

        if (
            not isnan(se_plot.to_numpy()).any()
            and check_is_sorted(se_plot.to_numpy())
            and (1 < unique(se_plot.to_numpy(), return_counts=True)[1]).all()
        ):

            print("Clustering within category...")

            vector = se_plot.to_numpy()

            matrix = df_plot.to_numpy()

            leaf_is = []

            for number in unique(vector):

                is_ = where(vector == number)[0]

                leaf_is.append(is_[cluster(matrix.T[is_])[0]])

            df_plot = df_plot.iloc[:, concatenate(leaf_is)]

            se_plot = se_plot.loc[df_plot.columns]

        n_row_plot = 1 + 1 + df_plot.shape[0]

        fraction_row = 1 / n_row_plot

        layout_base = {
            "height": max(480, 24 * n_row_plot),
            "width": 800,
            "margin": {"l": 200, "r": 200},
            "title": {"x": 0.5},
            "yaxis": {"domain": (0, 1 - 2 * fraction_row), "showticklabels": False},
            "yaxis2": {"domain": (1 - fraction_row, 1), "showticklabels": False},
            "annotations": [],
        }

        if layout is None:

            layout = layout_base

        else:

            layout = merge(layout_base, layout)

        annotation_base = {
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
                "text": "<b>{}</b>".format(se_plot.name),
                **annotation_base,
            }
        )

        y -= fraction_row

        for i, name in enumerate(("Score (\u0394)", "P-Value", "Q-Value")):

            layout["annotations"].append(
                {
                    "x": get_x(i),
                    "y": y,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(name),
                    **annotation_base,
                }
            )

        y -= fraction_row

        for label, (score, moe, p_value, q_value) in scores_plot.iterrows():

            layout["annotations"].append(
                {"x": 0, "y": y, "xanchor": "right", "text": label, **annotation_base}
            )

            for i, score_str in enumerate(
                (
                    "{:.2f} ({:.2f})".format(score, moe),
                    "{:.2e}".format(p_value),
                    "{:.2e}".format(q_value),
                )
            ):

                layout["annotations"].append(
                    {
                        "x": get_x(i),
                        "y": y,
                        "xanchor": "center",
                        "text": score_str,
                        **annotation_base,
                    }
                )

            y -= fraction_row

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = file_path.replace(".tsv", ".html")

        heatmap_trace_base = {
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
                        "x": se_plot.index.to_numpy(),
                        "z": se_plot.to_numpy().reshape((1, -1)),
                        "colorscale": DATA_TYPE_TO_COLORSCALE[se_data_type],
                        **heatmap_trace_base,
                    },
                    {
                        "yaxis": "y",
                        "x": df_plot.columns.to_numpy(),
                        "y": df_plot.index.to_numpy()[::-1],
                        "z": df_plot.to_numpy()[::-1],
                        "colorscale": DATA_TYPE_TO_COLORSCALE[df_data_type],
                        **heatmap_trace_base,
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return scores


def summarize(
    se,
    df_dicts,
    scores,
    plot_only_shared=False,
    se_ascending=True,
    se_data_type="continuous",
    plot_std=nan,
    html_file_path=None,
):

    if plot_only_shared:

        for df_dict in df_dicts.values():

            se = se.loc[se.index & df_dict["df"].columns]

    if se_ascending is not None:

        se.sort_values(ascending=se_ascending, inplace=True)

    se_plot = se.copy()

    if se_data_type == "continuous":

        se_plot = Series(
            ignore_nan_and_function_1(
                se_plot.to_numpy(), normalize, "-0-", update=True
            ).clip(min=-plot_std, max=plot_std),
            index=se_plot.index,
            name=se_plot.name,
        )

    n_space = 2

    n_row_plot = 1

    for df_dict in df_dicts.values():

        n_row_plot += n_space

        n_row_plot += df_dict["df"].shape[0]

    layout = {
        "height": max(480, 24 * n_row_plot),
        "width": 800,
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        "annotations": [],
    }

    fraction_row = 1 / n_row_plot

    yaxis = "yaxis{}".format(len(df_dicts) + 1)

    domain = 1 - fraction_row, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    heatmap_trace_base = {
        "type": "heatmap",
        "zmin": -plot_std,
        "zmax": plot_std,
        "showscale": False,
    }

    data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "x": se_plot.index.to_numpy(),
            "z": se_plot.to_numpy().reshape((1, -1)),
            "colorscale": DATA_TYPE_TO_COLORSCALE[se_data_type],
            **heatmap_trace_base,
        }
    ]

    annotation_base = {
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
            "text": "<b>{}</b>".format(se.name),
            **annotation_base,
        }
    )

    for df_i, (df_name, df_dict) in enumerate(df_dicts.items()):

        df_plot = df_dict["df"].reindex(columns=se_plot.index)

        scores_plot = scores[df_name].reindex(index=df_plot.index)

        if "emphasis" in df_dict:

            score_ascending = df_dict["emphasis"] == "-"

        else:

            score_ascending = False

        scores_plot.sort_values("Score", ascending=score_ascending, inplace=True)

        df_plot = df_plot.loc[scores_plot.index, :]

        if df_dict["data_type"] == "continuous":

            df_plot = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1,
                    1,
                    df_plot.to_numpy(),
                    normalize,
                    "-0-",
                    update=True,
                ).clip(min=-plot_std, max=plot_std),
                index=df_plot.index,
                columns=df_plot.columns,
            )

        yaxis = "yaxis{}".format(len(df_dicts) - df_i)

        domain = (
            max(0, domain[0] - fraction_row * (n_space + df_plot.shape[0])),
            domain[0] - fraction_row * n_space,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "x": df_plot.columns.to_numpy(),
                "y": df_plot.index.to_numpy()[::-1],
                "z": df_plot.to_numpy()[::-1],
                "colorscale": DATA_TYPE_TO_COLORSCALE[df_dict["data_type"]],
                **heatmap_trace_base,
            }
        )

        y = domain[1] + fraction_row / 2

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": y,
                "xanchor": "center",
                "text": "<b>{}</b>".format(df_name),
                **annotation_base,
            }
        )

        if df_i == 0:

            for score_i, score_str in enumerate(
                ("Score (\u0394)", "P-Value", "Q-Value")
            ):

                layout["annotations"].append(
                    {
                        "x": get_x(score_i),
                        "y": y,
                        "xanchor": "center",
                        "text": "<b>{}</b>".format(score_str),
                        **annotation_base,
                    }
                )

        y -= fraction_row

        for label, (score, moe, p_value, q_value) in scores_plot.iterrows():

            layout["annotations"].append(
                {"x": 0, "y": y, "xanchor": "right", "text": label, **annotation_base}
            )

            for score_i, score_str in enumerate(
                (
                    "{:.2f} ({:.2f})".format(score, moe),
                    "{:.2e}".format(p_value),
                    "{:.2e}".format(q_value),
                )
            ):

                layout["annotations"].append(
                    {
                        "x": get_x(score_i),
                        "y": y,
                        "xanchor": "center",
                        "text": score_str,
                        **annotation_base,
                    }
                )

            y -= fraction_row

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
