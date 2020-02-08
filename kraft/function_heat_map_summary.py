from numpy import apply_along_axis, nan
from pandas import DataFrame, Series

from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .ignore_nan_and_function_1 import ignore_nan_and_function_1
from .normalize import normalize
from .plot_plotly import plot_plotly


def function_heat_map_summary(
    vector,
    matrix_dicts,
    statistics,
    plot_only_shared=False,
    vector_ascending=True,
    vector_data_type="continuous",
    plot_std=nan,
    html_file_path=None,
):

    if plot_only_shared:

        for matrix_dict in matrix_dicts.values():

            vector = vector.loc[vector.index & matrix_dict["matrix"].columns]

    if vector_ascending is not None:

        vector.sort_values(ascending=vector_ascending, inplace=True)

    vector_ = vector.copy()

    if vector_data_type == "continuous":

        vector_ = Series(
            ignore_nan_and_function_1(vector_.values, normalize, "-0-"),
            name=vector_.name,
            index=vector_.index,
        ).clip(lower=-plot_std, upper=plot_std)

    n_space = 2

    n_row = 1

    for matrix_dict in matrix_dicts.values():

        n_row += n_space

        n_row += matrix_dict["matrix"].shape[0]

    layout = {
        "height": max(480, 24 * n_row),
        "width": 800,
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        "annotations": [],
    }

    fraction_row = 1 / n_row

    yaxis = "yaxis{}".format(len(matrix_dicts) + 1)

    domain = 1 - fraction_row, 1

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "type": "heatmap",
            "x": vector_.index,
            "z": vector_.to_frame().T,
            "colorscale": DATA_TYPE_COLORSCALE[vector_data_type],
            "showscale": False,
        }
    ]

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
            "text": "<b>{}</b>".format(vector.name),
            **annotation_,
        }
    )

    for matrix_index, (matrix_name, matrix_dict) in enumerate(matrix_dicts.items()):

        matrix_ = matrix_dict["matrix"].reindex(columns=vector_.index)

        statistics_ = statistics[matrix_name].reindex(index=matrix_.index)

        if "emphasis" in matrix_dict:

            score_ascending = matrix_dict["emphasis"] == "-"

        else:

            score_ascending = False

        statistics_.sort_values("Score", ascending=score_ascending, inplace=True)

        matrix_ = matrix_.loc[statistics_.index]

        if matrix_dict["data_type"] == "continuous":

            matrix_ = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1, 1, matrix_.values, normalize, "-0-"
                ),
                index=matrix_.index,
                columns=matrix_.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        yaxis = "yaxis{}".format(len(matrix_dicts) - matrix_index)

        domain = (
            max(0, domain[0] - fraction_row * (n_space + matrix_.shape[0])),
            domain[0] - fraction_row * n_space,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "type": "heatmap",
                "x": matrix_.columns,
                "y": matrix_.index[::-1],
                "z": matrix_.values[::-1],
                "colorscale": DATA_TYPE_COLORSCALE[matrix_dict["data_type"]],
                "showscale": False,
            }
        )

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": domain[1] + fraction_row / 2,
                "xanchor": "center",
                "text": "<b>{}</b>".format(matrix_name),
                **annotation_,
            }
        )

        def get_x(x_index):

            return 1.1 + x_index / 6.4

        if matrix_index == 0:

            for x_index, str_ in enumerate(("Score (\u0394)", "P-Value", "Q-Value")):

                layout["annotations"].append(
                    {
                        "x": get_x(x_index),
                        "y": 1 - fraction_row / 2,
                        "xanchor": "center",
                        "text": "<b>{}</b>".format(str_),
                        **annotation_,
                    }
                )

        y = domain[1] - fraction_row / 2

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

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
