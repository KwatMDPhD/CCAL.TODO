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
from pandas import DataFrame

from .array import check_is_sorted, get_not_nan_unique, ignore_nan_and_function_2
from .clustering import cluster
from .CONSTANT import RANDOM_SEED
from .dict_ import merge
from .plot import DATA_TYPE_TO_COLORSCALE, plot_plotly
from .significance import get_moe, get_p_values_and_q_values
from .table import get_extreme_label_, normalize as series_normalize

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


def _annotate_se(se, y):

    return {
        "x": 0,
        "y": y,
        "xanchor": "right",
        "text": "<b>{}</b>".format(se.name),
        **ANNOTATION_BASE,
    }


def _get_x(score_i):

    return 1.1 + score_i / 6.4


def _annotate_scores(scores, y, fraction_row, add_header):

    annotations = []

    if add_header:

        for i, text in enumerate(("Score (\u0394)", "P-Value", "Q-Value")):

            annotations.append(
                {
                    "x": _get_x(i),
                    "y": y,
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION_BASE,
                }
            )

    y -= fraction_row

    for text, (score, moe, p_value, q_value) in scores.iterrows():

        annotations.append(
            {"x": 0, "y": y, "xanchor": "right", "text": text, **ANNOTATION_BASE}
        )

        for i, text in enumerate(
            (
                "{:.2f} ({:.2f})".format(score, moe),
                "{:.2e}".format(p_value),
                "{:.2e}".format(q_value),
            )
        ):

            annotations.append(
                {
                    "x": _get_x(i),
                    "y": y,
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION_BASE,
                }
            )

        y -= fraction_row

    return annotations


def make(
    se,
    df,
    function_or_scores,
    se_ascending=True,
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
    title="Function Heat Map",
    directory_path=None,
):

    common_labels = se.index & df.columns

    assert 0 < common_labels.size

    print(
        "se.index ({}) and df.columns ({}) have {} in common.".format(
            se.index.size, df.columns.size, common_labels.size
        )
    )

    se = se.loc[common_labels]

    if se_ascending is not None:

        se.sort_values(ascending=se_ascending, inplace=True)

    df = df.loc[:, se.index]

    if callable(function_or_scores):

        function = function_or_scores

        scores = DataFrame(
            index=df.index, columns=("Score", "0.95 MoE", "P-Value", "Q-Value")
        )

        n_row, n_column = df.shape

        n_job = min(n_row, n_job)

        print("Computing scores (n_job={})...".format(n_job))

        pool = Pool(n_job)

        print("Score (function={})...".format(function.__name__))

        vector = se.to_numpy()

        matrix = df.to_numpy()

        seed(seed=random_seed)

        scores_ = asarray(
            pool.starmap(
                ignore_nan_and_function_2, ((vector, row, function) for row in matrix),
            )
        )

        scores.loc[:, "Score"] = scores_

        if 0 < n_sampling:

            print("0.95 MoE (n_sampling={})...".format(n_sampling))

            row_x_sampling = full((n_row, n_sampling), nan)

            n_sample = ceil(n_column * 0.632)

            def function_(numbers):

                return get_moe(numbers[~isnan(numbers)])

            for i in range(n_sampling):

                i_ = choice(n_column, size=n_sample)

                vector_ = vector[i_]

                row_x_sampling[:, i] = pool.starmap(
                    ignore_nan_and_function_2,
                    ((vector_, row, function) for row in matrix[:, i_]),
                )

            scores.loc[:, "0.95 MoE"] = apply_along_axis(function_, 1, row_x_sampling,)

        if 0 < n_permutation:

            print("P-Value and Q-Value (n_permutation={})...".format(n_permutation))

            row_x_permutation = full((n_row, n_permutation), nan)

            vector_ = vector.copy()

            for i in range(n_permutation):

                shuffle(vector_)

                row_x_permutation[:, i] = pool.starmap(
                    ignore_nan_and_function_2,
                    ((vector_, row, function) for row in matrix),
                )

            scores.loc[:, ("P-Value", "Q-Value")] = asarray(
                get_p_values_and_q_values(scores_, row_x_permutation.ravel(), "<>")
            ).T

        pool.terminate()

    else:

        scores = function_or_scores.reindex(labels=df.index)

    scores.sort_values("Score", ascending=score_ascending, inplace=True)

    if directory_path is not None:

        file_path = "{}scores.tsv".format(directory_path)

        scores.to_csv(file_path, sep="\t")

    if plot:

        plot_plotly(
            {
                "layout": {
                    "title": {"text": title},
                    "xaxis": {"title": {"text": "Rank"}},
                    "yaxis": {"title": {"text": "Score"}},
                },
                "data": [
                    {"name": name, "x": numbers.index, "y": numbers}
                    for name, numbers in scores.items()
                ],
            },
        )

        scores_plot = scores.copy()

        if n_extreme is not None:

            scores_plot = scores_plot.loc[
                get_extreme_label_(
                    scores_plot.loc[:, "Score"], "<>", n=n_extreme, plot=False
                )
            ].sort_values("Score", ascending=score_ascending)

        df = df.loc[scores_plot.index, :]

        if se_data_type == "continuous" and 1 < get_not_nan_unique(se.to_numpy()).size:

            print("Normalizing heat...")

            se = series_normalize(se, "-0-").clip(lower=-plot_std, upper=plot_std)

            se_min = -plot_std

            se_max = plot_std

        else:

            se_min = None

            se_max = None

        if df_data_type == "continuous" and all(
            1 < get_not_nan_unique(row).size for row in df.dropna(how="all").to_numpy()
        ):

            print("Normalizing heat...")

            df = dataframe_normalize(df, 1, "-0-").clip(lower=-plot_std, upper=plot_std)

            df_min = -plot_std

            df_max = plot_std

        else:

            df_min = None

            df_max = None

        vector = se.to_numpy()

        matrix = df.to_numpy()

        if all(
            (
                not isnan(vector).any(),
                not isnan(matrix).all(axis=1).any(),
                check_is_sorted(vector),
                (1 < unique(vector, return_counts=True)[1]).all(),
            )
        ):

            print("Clustering within category...")

            leaf_is = []

            for number in unique(vector):

                i_ = where(vector == number)[0]

                leaf_is.append(i_[cluster(matrix.T[i_])[0]])

            df = df.iloc[:, concatenate(leaf_is)]

            se = se.loc[df.columns]

        n_row = 1 + 1 + df.shape[0]

        fraction_row = 1 / n_row

        layout = merge(
            {
                "height": max(480, 24 * n_row),
                "yaxis": {"domain": (0, 1 - fraction_row * 2), "showticklabels": False},
                "yaxis2": {"domain": (1 - fraction_row, 1), "showticklabels": False},
                "title": {"text": title},
                "annotations": [_annotate_se(se, 1 - fraction_row / 2)],
            },
            LAYOUT_BASE,
        )

        layout["annotations"] += _annotate_scores(
            scores_plot, 1 - fraction_row / 2 * 3, fraction_row, True
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
                        "zmin": se_min,
                        "zmax": se_max,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[se_data_type],
                        **HEATMAP_BASE,
                    },
                    {
                        "yaxis": "y",
                        "x": df.columns.to_numpy(),
                        "y": df.index.to_numpy()[::-1],
                        "z": matrix[::-1],
                        "zmin": df_min,
                        "zmax": df_max,
                        "colorscale": DATA_TYPE_TO_COLORSCALE[df_data_type],
                        **HEATMAP_BASE,
                    },
                ],
            },
            file_path=file_path,
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
    title="Function Heat Map Summary",
    file_path=None,
):

    if plot_only_shared:

        for dict_ in df_dicts.values():

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

    n_space = 2

    n_row = 1

    for dict_ in df_dicts.values():

        n_row += n_space

        n_row += dict_["df"].shape[0]

    fraction_row = 1 / n_row

    layout = merge(
        {
            "height": max(480, 24 * n_row),
            "title": {"text": title},
            "annotations": [_annotate_se(se, 1 - fraction_row / 2)],
        },
        LAYOUT_BASE,
    )

    yaxis = "yaxis{}".format(len(df_dicts) + 1)

    domain = 1 - fraction_row, 1

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

    for i, (name, dict_) in enumerate(df_dicts.items()):

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

            df_min = -plot_std

            df_max = plot_std

        else:

            df_min = None

            df_max = None

        yaxis = "yaxis{}".format(len(df_dicts) - i)

        domain = (
            max(0, domain[0] - fraction_row * (n_space + df.shape[0])),
            domain[0] - fraction_row * n_space,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "x": df.columns.to_numpy(),
                "y": df.index.to_numpy()[::-1],
                "z": df.to_numpy()[::-1],
                "zmin": df_min,
                "zmax": df_max,
                "colorscale": DATA_TYPE_TO_COLORSCALE[dict_["data_type"]],
                **HEATMAP_BASE,
            }
        )

        y = domain[1] + fraction_row / 2

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": y,
                "xanchor": "center",
                "text": "<b>{}</b>".format(name),
                **ANNOTATION_BASE,
            }
        )

        layout["annotations"] += _annotate_scores(scores_, y, fraction_row, i == 0)

    plot_plotly({"layout": layout, "data": data}, file_path=file_path)
