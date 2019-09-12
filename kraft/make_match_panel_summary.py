from .ALMOST_ZERO import ALMOST_ZERO
from .guess_data_type import guess_data_type
from .make_match_panel_annotations import make_match_panel_annotations
from .normalize_dataframe import normalize_dataframe
from .normalize_series import normalize_series
from .plot_plotly_figure import plot_plotly_figure


def make_match_panel_summary(
    target,
    data_dicts,
    score_moe_p_value_fdr_dicts,
    plot_only_shared_by_target_and_all_data=False,
    target_ascending=True,
    score_ascending=False,
    target_data_type=None,
    plot_std=None,
    title=None,
    layout_side_margin=196,
    annotation_font_size=8.8,
    html_file_path=None,
):

    if plot_only_shared_by_target_and_all_data:

        for data_dict in data_dicts.values():

            target = target.loc[target.index & data_dict["dataframe"].columns]

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    target_to_plot = target.copy()

    if target_data_type is None:

        target_data_type = guess_data_type(target_to_plot)

    if target_data_type == "continuous":

        target_to_plot = normalize_series(target_to_plot, "-0-")

        if plot_std is not None:

            target_to_plot.clip(lower=-plot_std, upper=plot_std, inplace=True)

    n_row = 1 + len(data_dicts)

    for data_dict in data_dicts.values():

        n_row += data_dict["dataframe"].shape[0]

    layout = {
        "height": max(640, 32 * n_row),
        "margin": {"l": layout_side_margin, "r": layout_side_margin},
        "title": title,
        "xaxis": {"anchor": "y"},
        "annotations": [],
    }

    row_fraction = 1 / n_row

    yaxis_name = "yaxis{}".format(len(data_dicts) + 1)

    domain_end = 1

    domain_start = domain_end - row_fraction

    if abs(domain_start) <= ALMOST_ZERO:

        domain_start = 0

    annotation_font = {"size": annotation_font_size}

    layout[yaxis_name] = {
        "domain": (domain_start, domain_end),
        "tickmode": "array",
        "tickvals": (0,),
        "ticktext": (target_to_plot.name,),
        "tickfont": annotation_font,
    }

    figure_data = [
        {
            "yaxis": yaxis_name.replace("axis", ""),
            "type": "heatmap",
            "x": target_to_plot.index,
            "z": target_to_plot.to_frame().T,
            "colorscale": make_colorscale_from_colors(
                pick_colors(target_to_plot, data_type=target_data_type)
            ),
            "showscale": False,
        }
    ]

    for data_index, (data_name, data_dict) in enumerate(data_dicts.items()):

        print("Making match panel for {}...".format(data_name))

        data_to_plot = data_dict["dataframe"].reindex(columns=target_to_plot.index)

        score_moe_p_value_fdr_to_plot = score_moe_p_value_fdr_dicts[data_name].reindex(
            index=data_to_plot.index
        )

        score_moe_p_value_fdr_to_plot.sort_values(
            "Score", ascending=score_ascending, inplace=True
        )

        data_to_plot = data_to_plot.loc[score_moe_p_value_fdr_to_plot.index]

        if data_dict["type"] == "continuous":

            data_to_plot = normalize_dataframe(data_to_plot, 1, "-0-")

            if plot_std is not None:

                data_to_plot.clip(lower=-plot_std, upper=plot_std, inplace=True)

        yaxis_name = "yaxis{}".format(len(data_dicts) - data_index)

        domain_end = domain_start - row_fraction

        if abs(domain_end) <= ALMOST_ZERO:

            domain_end = 0

        domain_start = domain_end - data_to_plot.shape[0] * row_fraction

        if abs(domain_start) <= ALMOST_ZERO:

            domain_start = 0

        layout[yaxis_name] = {
            "domain": (domain_start, domain_end),
            "dtick": 1,
            "tickfont": annotation_font,
        }

        figure_data.append(
            {
                "yaxis": yaxis_name.replace("axis", ""),
                "type": "heatmap",
                "x": data_to_plot.columns,
                "y": data_to_plot.index[::-1],
                "z": data_to_plot.values[::-1],
                "colorscale": make_colorscale_from_colors(
                    pick_colors(data_to_plot, data_type=data_dict["type"])
                ),
                "showscale": False,
            }
        )

        layout_annotation_template = {
            "xref": "paper",
            "yref": "paper",
            "yanchor": "middle",
            "font": annotation_font,
            "showarrow": False,
        }

        layout["annotations"].append(
            {
                "xanchor": "center",
                "x": 0.5,
                "y": domain_end + (row_fraction / 2),
                "text": "<b>{}</b>".format(data_name),
                **layout_annotation_template,
            }
        )

        layout_annotation_template.update({"xanchor": "left", "width": 64})

        for i, (annotation, strs) in enumerate(
            make_match_panel_annotations(score_moe_p_value_fdr_to_plot).items()
        ):

            x = 1.0016 + i / 10

            if data_index == 0:

                layout["annotations"].append(
                    {
                        "x": x,
                        "y": 1 - (row_fraction / 2),
                        "text": "<b>{}</b>".format(annotation),
                        **layout_annotation_template,
                    }
                )

            y = domain_end - (row_fraction / 2)

            for str_ in strs:

                layout["annotations"].append(
                    {
                        "x": x,
                        "y": y,
                        "text": "<b>{}</b>".format(str_),
                        **layout_annotation_template,
                    }
                )

                y -= row_fraction

    plot_plotly_figure({"layout": layout, "data": figure_data}, html_file_path)
