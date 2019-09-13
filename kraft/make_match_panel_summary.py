from numpy import nan

from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .make_match_panel_annotations import make_match_panel_annotations
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .normalize_dataframe import normalize_dataframe
from .normalize_series import normalize_series
from .plot_plotly_figure import plot_plotly_figure


def make_match_panel_summary(
    target,
    data_dicts,
    score_moe_p_value_fdr_dicts,
    plot_only_shared_by_target_and_all_data=False,
    target_ascending=True,
    target_data_type="continuous",
    data_data_type="continuous",
    plot_std=nan,
    layout=None,
    html_file_path=None,
):

    if plot_only_shared_by_target_and_all_data:

        for data_dict in data_dicts.values():

            target = target.loc[target.index & data_dict["dataframe"].columns]

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    target_to_plot = target.copy()

    if target_data_type == "continuous":

        target_to_plot = normalize_series(target_to_plot, "-0-").clip(
            lower=-plot_std, upper=plot_std
        )

        target_to_plot_z_magnitude = plot_std

    else:

        target_to_plot_z_magnitude = nan

    n_space = 2

    n_row = 1

    for data_dict in data_dicts.values():

        n_row += n_space

        n_row += data_dict["dataframe"].shape[0]

    layout_template = {
        "height": max(500, 25 * n_row),
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        "xaxis": {"showticklabels": False},
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

    row_fraction = 1 / n_row

    yaxis = "yaxis{}".format(len(data_dicts) + 1)

    domain = (1 - row_fraction, 1)

    layout[yaxis] = {"domain": domain, "showticklabels": False}

    figure_data = [
        {
            "yaxis": yaxis.replace("axis", ""),
            "type": "heatmap",
            "name": "Target",
            "x": target_to_plot.index,
            "z": target_to_plot.to_frame().T,
            "zmin": -target_to_plot_z_magnitude,
            "zmax": target_to_plot_z_magnitude,
            "colorscale": DATA_TYPE_COLORSCALE[target_data_type],
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
            "text": "<b>{}</b>".format(target.name),
            **layout_annotation_template,
        }
    )

    for data_index, (data_name, data_dict) in enumerate(data_dicts.items()):

        print("Making match panel with data {}...".format(data_name))

        data_to_plot = data_dict["dataframe"].reindex(columns=target_to_plot.index)

        score_moe_p_value_fdr_to_plot = score_moe_p_value_fdr_dicts[data_name].reindex(
            index=data_to_plot.index
        )

        if "emphasis" in data_dict:

            score_ascending = data_dict["emphasis"] == "-"

        else:

            score_ascending = False

        score_moe_p_value_fdr_to_plot.sort_values(
            "Score", ascending=score_ascending, inplace=True
        )

        data_to_plot = data_to_plot.loc[score_moe_p_value_fdr_to_plot.index]

        if data_data_type == "continuous":

            data_to_plot = normalize_dataframe(data_to_plot, 1, "-0-").clip(
                lower=-plot_std, upper=plot_std
            )

            data_to_plot_z_magnitude = plot_std

        else:

            data_to_plot_z_magnitude = nan

        yaxis = "yaxis{}".format(len(data_dicts) - data_index)

        domain = (
            max(0, domain[0] - row_fraction * (n_space + data_to_plot.shape[0])),
            domain[0] - row_fraction * n_space,
        )

        layout[yaxis] = {"domain": domain, "showticklabels": False}

        figure_data.append(
            {
                "yaxis": yaxis.replace("axis", ""),
                "type": "heatmap",
                "name": "Data {}".format(data_name),
                "x": data_to_plot.columns,
                "y": data_to_plot.index[::-1],
                "z": data_to_plot.values[::-1],
                "zmin": -data_to_plot_z_magnitude,
                "zmax": data_to_plot_z_magnitude,
                "colorscale": DATA_TYPE_COLORSCALE[data_dict["data_type"]],
                "showscale": False,
            }
        )

        layout["annotations"].append(
            {
                "x": 0.5,
                "y": domain[1] + (row_fraction / 2),
                "xanchor": "center",
                "text": "<b>{}</b>".format(data_name),
                **layout_annotation_template,
            }
        )

        for annotation_index, (annotation, annotation_values) in enumerate(
            make_match_panel_annotations(score_moe_p_value_fdr_to_plot).items()
        ):

            x = 1.1 + annotation_index / 5

            if data_index == 0:

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

    plot_plotly_figure({"layout": layout, "data": figure_data}, html_file_path)
