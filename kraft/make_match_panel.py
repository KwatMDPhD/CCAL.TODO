from numpy import nan

from .apply_function_on_vector_and_each_matrix_row_and_compute_statistics import (
    apply_function_on_vector_and_each_matrix_row_and_compute_statistics,
)
from .cluster_matrix import cluster_matrix
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .is_sorted_array import is_sorted_array
from .make_match_panel_annotations import make_match_panel_annotations
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .normalize_dataframe import normalize_dataframe
from .normalize_series import normalize_series
from .plot_plotly_figure import plot_plotly_figure
from .select_series_indices import select_series_indices


def make_match_panel(
    target,
    data,
    target_ascending=True,
    score_moe_p_value_fdr=None,
    score_ascending=False,
    n_extreme=8,
    fraction_extreme=None,
    plot=True,
    target_data_type="continuous",
    data_data_type="continuous",
    plot_std=nan,
    cluster_within_category=True,
    layout=None,
    file_path_prefix=None,
    **apply_function_on_vector_and_each_matrix_row_and_compute_statistics_keyword_arguments,
):

    common_indices = target.index & data.columns

    print(
        "target.index ({}) and data.columns ({}) have {} in common.".format(
            target.index.size, data.columns.size, common_indices.size
        )
    )

    target = target[common_indices]

    if target_ascending is not None:

        target.sort_values(ascending=target_ascending, inplace=True)

    data = data[target.index]

    if score_moe_p_value_fdr is None:

        score_moe_p_value_fdr = apply_function_on_vector_and_each_matrix_row_and_compute_statistics(
            target.values,
            data.values,
            n_extreme=n_extreme,
            fraction_extreme=fraction_extreme,
            **apply_function_on_vector_and_each_matrix_row_and_compute_statistics_keyword_arguments,
        )

        score_moe_p_value_fdr.index = data.index

    else:

        score_moe_p_value_fdr = score_moe_p_value_fdr.reindex(
            data_to_plot_index=data.index
        )

    score_moe_p_value_fdr.sort_values("Score", ascending=score_ascending, inplace=True)

    if file_path_prefix is not None:

        score_moe_p_value_fdr.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    if not plot:

        return score_moe_p_value_fdr

    print("Plotting...")

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.statistics.html".format(file_path_prefix)

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "Statistics"},
                "xaxis": {"title": {"text": "Rank"}, "automargin": True},
                "yaxis": {"title": {"text": "Score"}},
            },
            "data": [
                {
                    "type": "scatter",
                    "name": name,
                    "x": score_moe_p_value_fdr.index,
                    "y": score_moe_p_value_fdr[name].values,
                }
                for name in score_moe_p_value_fdr.columns
            ],
        },
        html_file_path,
    )

    target_to_plot = target.copy()

    score_moe_p_value_fdr_to_plot = score_moe_p_value_fdr.copy()

    if n_extreme is not None or fraction_extreme is not None:

        score_moe_p_value_fdr_to_plot = score_moe_p_value_fdr_to_plot.loc[
            select_series_indices(
                score_moe_p_value_fdr_to_plot["Score"],
                "<>",
                n=n_extreme,
                fraction=fraction_extreme,
                plot=False,
            )
        ]

    score_moe_p_value_fdr_to_plot.sort_values(
        "Score", ascending=score_ascending, inplace=True
    )

    data_to_plot = data.loc[score_moe_p_value_fdr_to_plot.index]

    if target_data_type == "continuous":

        target_to_plot = normalize_series(target_to_plot, "-0-").clip(
            lower=-plot_std, upper=plot_std
        )

        target_to_plot_z_magnitude = plot_std

    else:

        target_to_plot_z_magnitude = nan

    if data_data_type == "continuous":

        data_to_plot = normalize_dataframe(data_to_plot, 1, "-0-").clip(
            lower=-plot_std, upper=plot_std
        )

        data_to_plot_z_magnitude = plot_std

    else:

        data_to_plot_z_magnitude = nan

    if (
        cluster_within_category
        and not target_to_plot.isna().any()
        and is_sorted_array(target_to_plot.values)
        and (1 < target_to_plot.value_counts()).all()
    ):

        print("Clustering within category...")

        data_to_plot = data_to_plot.iloc[
            :,
            cluster_matrix(
                data_to_plot.values,
                1,
                groups=target_to_plot.values,
                raise_for_bad=False,
            ),
        ]

        target_to_plot = target_to_plot[data_to_plot.columns]

    n_row = 1 + 1 + data_to_plot.shape[0]

    row_fraction = 1 / n_row

    layout_template = {
        "height": max(500, 25 * n_row),
        "width": 1000,
        "margin": {"l": 200, "r": 200},
        "title": {"x": 0.5},
        "xaxis": {"showticklabels": False},
        "yaxis": {"domain": (0, 1 - 2 * row_fraction), "showticklabels": False},
        "yaxis2": {"domain": (1 - row_fraction, 1), "showticklabels": False},
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts_recursively(layout_template, layout)

    figure_data = [
        {
            "yaxis": "y2",
            "type": "heatmap",
            "name": "Target",
            "x": target_to_plot.index,
            "z": target_to_plot.to_frame().T,
            "zmin": -target_to_plot_z_magnitude,
            "zmax": target_to_plot_z_magnitude,
            "colorscale": DATA_TYPE_COLORSCALE[target_data_type],
            "showscale": False,
        },
        {
            "yaxis": "y",
            "type": "heatmap",
            "name": "Data",
            "x": data_to_plot.columns,
            "y": data_to_plot.index[::-1],
            "z": data_to_plot.iloc[::-1],
            "zmin": -data_to_plot_z_magnitude,
            "zmax": data_to_plot_z_magnitude,
            "colorscale": DATA_TYPE_COLORSCALE[data_data_type],
            "showscale": False,
        },
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

    for annotation_index, (annotation, annotation_values) in enumerate(
        make_match_panel_annotations(score_moe_p_value_fdr_to_plot).items()
    ):

        x = 1.1 + annotation_index / 5

        layout["annotations"].append(
            {
                "x": x,
                "y": 1 - (row_fraction / 2),
                "xanchor": "center",
                "text": "<b>{}</b>".format(annotation),
                **layout_annotation_template,
            }
        )

        y = layout["yaxis"]["domain"][1] - (row_fraction / 2)

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

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.html".format(file_path_prefix)

    plot_plotly_figure({"layout": layout, "data": figure_data}, html_file_path)

    return score_moe_p_value_fdr
