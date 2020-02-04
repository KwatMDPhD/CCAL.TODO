from numpy import nan
from pandas import DataFrame, Series

from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .ignore_nan_and_function_1 import ignore_nan_and_function_1
from .merge_2_dicts import merge_2_dicts
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
    layout=None,
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

    layout_template = {
        "height": max(500, 25 * n_row),
        "width": 800,
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts(layout_template, layout)

    row_fraction = 1 / n_row

    yaxis = "yaxis{}".format(len(matrix_dicts) + 1)

    domain = (1 - row_fraction, 1)

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    figure_data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "type": "heatmap",
            "name": "Target",
            "x": vector_.index,
            "z": vector_.to_frame().T,
            "colorscale": DATA_TYPE_COLORSCALE[vector_data_type],
            "showscale": False,
        }
    ]

    layout_annotation_template = {
        "xref": "paper",
        "yref": "paper",
        "yanchor": "middle",
        "font": {"size": 10},
        "showarrow": False,
    }

    layout["annotations"].append(
        {
            "x": 0,
            "y": 1 - (row_fraction / 2),
            "xanchor": "right",
            "text": "<b>{}</b>".format(vector.name),
            **layout_annotation_template,
        }
    )

    for matrix_index, (matrix_name, matrix_dict) in enumerate(matrix_dicts.items()):

        print("Making match panel with data {}...".format(matrix_name))

        data_to_plot = matrix_dict["dataframe"].reindex(columns=vector_.index)

        score_moe_p_value_fdr_to_plot = statistics[matrix_name].reindex(
            index=data_to_plot.index
        )

        if "emphasis" in matrix_dict:

            score_ascending = matrix_dict["emphasis"] == "-"

        else:

            score_ascending = False

        score_moe_p_value_fdr_to_plot.sort_values(
            "Score", ascending=score_ascending, inplace=True
        )

        data_to_plot = data_to_plot.loc[score_moe_p_value_fdr_to_plot.index]

        if matrix_dict["matrix_data_type"] == "continuous":

            matrix_ = DataFrame(
                apply_along_axis(
                    ignore_nan_and_function_1, 1, matrix_.values, normalize, "-0-"
                ),
                index=matrix_.index,
                columns=matrix_.columns,
            ).clip(lower=-plot_std, upper=plot_std)

        yaxis = "yaxis{}".format(len(matrix_dicts) - matrix_index)

        domain = (
            max(0, domain[0] - row_fraction * (n_space + data_to_plot.shape[0])),
            domain[0] - row_fraction * n_space,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        figure_data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "type": "heatmap",
                "name": matrix_name,
                "x": data_to_plot.columns,
                "y": data_to_plot.index[::-1],
                "z": data_to_plot.values[::-1],
                "colorscale": DATA_TYPE_COLORSCALE[matrix_dict["data_type"]],
                "showscale": False,
            }
        )

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": domain[1] + (row_fraction / 2),
                "xanchor": "center",
                "text": "<b>{}</b>".format(matrix_name),
                **layout_annotation_template,
            }
        )

        for annotation_index, (annotation, annotation_values) in enumerate(
            make_match_panel_annotations(score_moe_p_value_fdr_to_plot).items()
        ):

            x = 1.1 + annotation_index / 6.4

            if matrix_index == 0:

                layout["annotations"].append(
                    {
                        "x": x,
                        "y": 1 - (row_fraction / 2),
                        "xanchor": "center",
                        "text": "<b>{}</b>".format(annotation),
                        **layout_annotation_template,
                    }
                )

            y = domain[1] - (row_fraction / 2)

            for data_to_plot_index, annotation in zip(
                data_to_plot.index, annotation_values
            ):

                layout["annotations"].append(
                    {
                        "x": 0,
                        "y": y,
                        "xanchor": "right",
                        "text": data_to_plot_index,
                        **layout_annotation_template,
                    }
                )

                layout["annotations"].append(
                    {
                        "x": x,
                        "y": y,
                        "xanchor": "center",
                        "text": annotation,
                        **layout_annotation_template,
                    }
                )

                y -= row_fraction

    plot_plotly({"layout": layout, "data": figure_data}, html_file_path)
