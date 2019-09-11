from .apply_function_on_vector_and_each_matrix_row_and_compute_statistics import (
    apply_function_on_vector_and_each_matrix_row_and_compute_statistics,
)
from .cluster_matrix import cluster_matrix
from .compute_information_coefficient_between_2_vectors import (
    compute_information_coefficient_between_2_vectors,
)
from .get_data_type import get_data_type
from .is_sorted_array import is_sorted_array
from .make_colorscale_from_colors import make_colorscale_from_colors
from .make_match_panel_annotations import make_match_panel_annotations
from .normalize_dataframe import normalize_dataframe
from .normalize_series import normalize_series
from .pick_colors import pick_colors
from .plot_plotly_figure import plot_plotly_figure
from .RANDOM_SEED import RANDOM_SEED
from .select_series_indices import select_series_indices


def make_match_panel(
    target,
    data,
    target_ascending=True,
    score_moe_p_value_fdr=None,
    n_job=1,
    match_function=compute_information_coefficient_between_2_vectors,
    n_required_for_match_function=2,
    raise_for_n_less_than_required=False,
    n_extreme=8,
    fraction_extreme=None,
    random_seed=RANDOM_SEED,
    n_sampling=10,
    n_permutation=10,
    score_ascending=False,
    plot=True,
    cluster_within_category=True,
    target_type=None,
    data_type=None,
    plot_std=None,
    title=None,
    layout_side_margin=196,
    annotation_font_size=8.8,
    file_path_prefix=None,
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
            n_job,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
            n_extreme,
            fraction_extreme,
            random_seed,
            n_sampling,
            n_permutation,
        )

        score_moe_p_value_fdr.index = data.index

    else:

        score_moe_p_value_fdr = score_moe_p_value_fdr.reindex(index=data.index)

    if score_moe_p_value_fdr.isna().values.all():

        print("score_moe_p_value_fdr has only NA.")

        return score_moe_p_value_fdr

    score_moe_p_value_fdr.sort_values("Score", ascending=score_ascending, inplace=True)

    if file_path_prefix is not None:

        score_moe_p_value_fdr.to_csv("{}.tsv".format(file_path_prefix), sep="\t")

    if not plot:

        return score_moe_p_value_fdr

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.statistics.html".format(file_path_prefix)

    plot_plotly_figure(
        {
            "layout": {
                "title": {"text": "Statistics"},
                "xaxis": {"title": {"text": "Rank"}},
                "yaxis": {
                    "title": {"text": "Score<br>{}".format(match_function.__name__)}
                },
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
        ].sort_values("Score", ascending=score_ascending)

    data_to_plot = data.loc[score_moe_p_value_fdr_to_plot.index]

    target_to_plot = target.copy()

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

    if target_type is None:

        target_type = get_data_type(target_to_plot)

    if target_type == "continuous":

        target_to_plot = normalize_series(target_to_plot, "-0-")

        if plot_std is not None:

            target_to_plot.clip(lower=-plot_std, upper=plot_std, inplace=True)

    if data_type is None:

        data_type = get_data_type(data_to_plot)

    if data_type == "continuous":

        data_to_plot = normalize_dataframe(data_to_plot, 1, "-0-")

        if plot_std is not None:

            data_to_plot.clip(lower=-plot_std, upper=plot_std, inplace=True)

    row_fraction = max(0.01, 1 / (data_to_plot.shape[0] + 2))

    target_yaxis_domain = (1 - row_fraction, 1)

    data_yaxis_domain = (0, 1 - 2 * row_fraction)

    data_row_fraction = (
        data_yaxis_domain[1] - data_yaxis_domain[0]
    ) / data_to_plot.shape[0]

    annotation_font = {"size": annotation_font_size}

    layout = {
        "height": max(640, 32 * (data_to_plot.shape[0] + 2)),
        "margin": {"l": layout_side_margin, "r": layout_side_margin},
        "title": title,
        "xaxis": {"anchor": "y", "tickfont": annotation_font},
        "yaxis": {"domain": data_yaxis_domain, "dtick": 1, "tickfont": annotation_font},
        "yaxis2": {
            "domain": target_yaxis_domain,
            "tickmode": "array",
            "tickvals": (0,),
            "ticktext": (target_to_plot.name,),
            "tickfont": annotation_font,
        },
        "annotations": [],
    }

    figure_data = [
        {
            "yaxis": "y2",
            "type": "heatmap",
            "x": target_to_plot.index,
            "z": target_to_plot.to_frame().T,
            "colorscale": make_colorscale_from_colors(
                pick_colors(target_to_plot, data_type=target_type)
            ),
            "showscale": False,
        },
        {
            "yaxis": "y",
            "type": "heatmap",
            "x": data_to_plot.columns,
            "y": data_to_plot.index[::-1],
            "z": data_to_plot.iloc[::-1],
            "colorscale": make_colorscale_from_colors(
                pick_colors(data_to_plot, data_type=data_type)
            ),
            "showscale": False,
        },
    ]

    layout_annotation_template = {
        "xref": "paper",
        "yref": "paper",
        "xanchor": "left",
        "yanchor": "middle",
        "font": annotation_font,
        "width": 64,
        "showarrow": False,
    }

    for i, (annotation, strs) in enumerate(
        make_match_panel_annotations(score_moe_p_value_fdr_to_plot).items()
    ):

        x = 1.0016 + i / 10

        layout["annotations"].append(
            {
                "x": x,
                "y": 1 - (row_fraction / 2),
                "text": "<b>{}</b>".format(annotation),
                **layout_annotation_template,
            }
        )

        y = data_yaxis_domain[1] - (data_row_fraction / 2)

        for str_ in strs:

            layout["annotations"].append(
                {
                    "x": x,
                    "y": y,
                    "text": "<b>{}</b>".format(str_),
                    **layout_annotation_template,
                }
            )

            y -= data_row_fraction

    if file_path_prefix is None:

        html_file_path = None

    else:

        html_file_path = "{}.html".format(file_path_prefix)

    plot_plotly_figure({"layout": layout, "data": figure_data}, html_file_path)

    return score_moe_p_value_fdr
