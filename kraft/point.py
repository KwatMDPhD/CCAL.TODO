from numpy import apply_along_axis, full, isnan, nan, unique
from pandas import DataFrame
from sklearn.manifold import MDS

from .array import normalize
from .CONSTANT import RANDOM_SEED
from .geometry import get_convex_hull, get_triangulation
from .plot import COLORBAR, get_color, make_colorscale, plot_plotly
from .support import merge_2_dicts


def pull_point(node_x_dimension, point_x_node):

    point_x_dimension = full((point_x_node.shape[0], node_x_dimension.shape[1]), nan)

    for point_index in range(point_x_node.shape[0]):

        pulls = point_x_node[point_index, :]

        for dimension_index in range(node_x_dimension.shape[1]):

            point_x_dimension[point_index, dimension_index] = (
                pulls * node_x_dimension[:, dimension_index]
            ).sum() / pulls.sum()

    return point_x_dimension


def map_point(
    point_x_point_distance,
    n_dimension,
    metric=True,
    n_init=int(1e3),
    max_iter=int(1e3),
    verbose=0,
    eps=1e-3,
    n_job=1,
    random_seed=RANDOM_SEED,
):

    point_x_dimension = MDS(
        n_components=n_dimension,
        metric=metric,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        n_jobs=n_job,
        random_state=random_seed,
        dissimilarity="precomputed",
    ).fit_transform(point_x_point_distance)

    return apply_along_axis(normalize, 0, point_x_dimension, "0-1")


def plot_node_point(
    node_x_dimension,
    point_x_dimension,
    node_marker_size=16,
    opacity=0.8,
    # TODO: consider using vector
    point_group=None,
    GRID=None,
    grid_point_probability=None,
    grid_point_group=None,
    point_group_colorscale=None,
    point_value=None,
    point_value_na_opacity=None,
    point_value_colorscale=None,
    ticktext_function=None,
    layout=None,
    show_node_text=True,
    point_trace=None,
    points_to_highlight=(),
    html_file_path=None,
):

    node_x_dimension = DataFrame(
        {
            "x": node_x_dimension.iloc[:, 1].values,
            "y": 1 - node_x_dimension.iloc[:, 0].values,
        },
        index=node_x_dimension.index,
    )

    point_x_dimension = DataFrame(
        {
            "x": point_x_dimension.iloc[:, 1].values,
            "y": 1 - point_x_dimension.iloc[:, 0].values,
        },
        index=point_x_dimension.index,
    )

    title_text = "{} {} and {} {}".format(
        node_x_dimension.index.size,
        node_x_dimension.index.name,
        point_x_dimension.index.size,
        point_x_dimension.index.name,
    )

    if point_value is not None:

        title_text = "{}<br>{}".format(title_text, point_value.name)

    axis = {"showgrid": False, "zeroline": False, "showticklabels": False}

    layout_template = {
        "height": 880,
        "width": 880,
        "title": {
            "x": 0.5,
            "text": "<b>{}</b>".format(title_text),
            "font": {
                "size": 24,
                "color": "#4e40d8",
                "family": "Times New Roman, sans-serif",
            },
        },
        "xaxis": axis,
        "yaxis": axis,
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge_2_dicts(layout_template, layout)

    triangulation_xs, triangulation_ys = get_triangulation(node_x_dimension)

    convex_hull_xs, convex_hull_ys = get_convex_hull(node_x_dimension)

    data = [
        {
            "name": "Line",
            "x": triangulation_xs + convex_hull_xs,
            "y": triangulation_ys + convex_hull_ys,
            "mode": "lines",
            "line": {"color": "#171412"},
        }
    ]

    data.append(
        {
            "name": node_x_dimension.index.name,
            "x": node_x_dimension["x"],
            "y": node_x_dimension["y"],
            "text": node_x_dimension.index,
            "mode": "markers",
            "marker": {
                "size": node_marker_size,
                "color": "#23191e",
                "line": {"width": 1, "color": "#ebf6f7"},
            },
            "hoverinfo": "text",
        }
    )

    if show_node_text:

        border_arrow_width = 1.6

        border_arrow_color = "#ebf6f7"

        layout["annotations"] += [
            {
                "x": x,
                "y": y,
                "text": "<b>{}</b>".format(node_name),
                "font": {
                    "size": 16,
                    "color": "#23191e",
                    "family": "Gravitas One, monospace",
                },
                "bgcolor": "#ffffff",
                "borderpad": 2,
                "borderwidth": border_arrow_width,
                "bordercolor": border_arrow_color,
                "arrowwidth": border_arrow_width,
                "arrowcolor": border_arrow_color,
                "opacity": opacity,
            }
            for node_name, (x, y) in node_x_dimension.iterrows()
        ]

    point_trace_template = {
        "name": point_x_dimension.index.name,
        "mode": "markers",
        "marker": {
            "size": 16,
            "color": "#20d9ba",
            "line": {"width": 0.8, "color": "#ebf6f7"},
            "opacity": opacity,
        },
        "hoverinfo": "text",
    }

    if point_trace is None:

        point_trace = point_trace_template

    else:

        point_trace = merge_2_dicts(point_trace_template, point_trace)

    if grid_point_group is not None:

        grid_group_not_nan_unique = unique(grid_point_group[~isnan(grid_point_group)])

        # TODO: consider renaming group to reflect int
        group_color = {
            group: get_color(
                point_group_colorscale, group, grid_group_not_nan_unique.size
            )
            for group in grid_group_not_nan_unique
        }

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": GRID,
                "y": 1 - GRID,
                "z": grid_point_probability,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
            }
        )

        for group in grid_group_not_nan_unique:

            z = grid_point_probability.copy()

            z[grid_point_group != group] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": GRID,
                    "y": 1 - GRID,
                    "z": z,
                    "colorscale": make_colorscale(
                        ("rgb(255, 255, 255)", group_color[group])
                    ),
                    "showscale": False,
                    "opacity": opacity,
                    "hoverinfo": "none",
                }
            )

    if point_value is not None:

        point_value = point_value.reindex(index=point_x_dimension.index)

        point_value_opacity = opacity

        if point_value_na_opacity is None:

            point_value.dropna(inplace=True)

        else:

            point_value_opacity = point_value.where(
                isnan, other=point_value_opacity
            ).fillna(value=point_value_na_opacity)

        point_value = point_value[
            point_value.abs().sort_values(na_position="first").index
        ]

        point_x_dimension = point_x_dimension.loc[point_value.index]

        if point_value.astype(float).map(float.is_integer).all():

            tickvals = point_value.unique()

            if ticktext_function is None:

                ticktext_function = "{:.0f}".format

        else:

            tickvals = point_value.describe()[
                ["min", "25%", "50%", "mean", "75%", "max"]
            ].values

            ticktext_function = "{:.2e}".format

        data.append(
            merge_2_dicts(
                point_trace,
                {
                    "x": point_x_dimension["x"],
                    "y": point_x_dimension["y"],
                    "text": tuple(
                        "{}<br>{:.2e}".format(point_name, value)
                        for point_name, value in point_value.items()
                    ),
                    "marker": {
                        "color": point_value,
                        "colorscale": point_value_colorscale,
                        "colorbar": merge_2_dicts(
                            COLORBAR,
                            {
                                "tickmode": "array",
                                "tickvals": tickvals,
                                "ticktext": tuple(
                                    ticktext_function(tickval) for tickval in tickvals
                                ),
                            },
                        ),
                        "opacity": point_value_opacity,
                    },
                },
            )
        )

    elif point_group is not None:

        for group in grid_group_not_nan_unique:

            is_group = point_group == group

            name = "Label {:.0f}".format(group)

            data.append(
                merge_2_dicts(
                    point_trace,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": point_x_dimension["x"][is_group],
                        "y": point_x_dimension["y"][is_group],
                        "text": point_x_dimension.index[is_group],
                        "marker": {"color": group_color[group]},
                    },
                )
            )

    else:

        data.append(
            merge_2_dicts(
                point_trace,
                {
                    "x": point_x_dimension["x"],
                    "y": point_x_dimension["y"],
                    "text": point_x_dimension.index,
                },
            )
        )

    layout["annotations"] += [
        {
            "x": point_x_dimension.loc[point, "x"],
            "y": point_x_dimension.loc[point, "y"],
            "text": "<b>{}</b>".format(point),
            "arrowhead": 2,
            "arrowwidth": 2,
            "arrowcolor": "#c93756",
            "standoff": None,
            "clicktoshow": "onoff",
        }
        for point in points_to_highlight
    ]

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
