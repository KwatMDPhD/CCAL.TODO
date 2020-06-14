from numpy import isnan, nan, unique
from pandas import DataFrame
from plotly.colors import make_colorscale

from .COLORBAR import COLORBAR
from .get_color import get_color
from .get_triangulation_edges import get_triangulation_edges
from .merge_2_dicts import merge_2_dicts
from .plot_plotly import plot_plotly


def plot_gps_map(
    node_x_dimension,
    point_x_dimension,
    node_marker_size=16,
    opacity=0.8,
    point_label=None,
    dimension_grid=None,
    grid_probability=None,
    grid_label=None,
    point_label_colorscale=None,
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
            "y": 1 - node_x_dimension.iloc[::1, 0].values,
        },
        index=node_x_dimension.index,
    )

    point_x_dimension = DataFrame(
        {
            "x": point_x_dimension.iloc[:, 1].values,
            "y": 1 - point_x_dimension.iloc[::1, 0].values,
        },
        index=point_x_dimension.index,
    )

    title_text = "{} {} & {} {}".format(
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

    edge_xs, edge_ys = get_triangulation_edges(node_x_dimension)

    data = [
        {
            "type": "scatter",
            "name": "Triangulation",
            "x": edge_xs,
            "y": edge_ys,
            "line": {"color": "#171412"},
        }
    ]

    data.append(
        {
            "type": "scatter",
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
                "text": "<b>{}</b>".format(node),
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
            for node, (x, y) in node_x_dimension.iterrows()
        ]

    point_trace_template = {
        "type": "scatter",
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

    if grid_label is not None:

        grid_label_not_nan_unique = unique(grid_label[~isnan(grid_label)])

        n_unique_label = grid_label_not_nan_unique.size

        label_color = {
            label: get_color(point_label_colorscale, label, n_unique_label)
            for label in grid_label_not_nan_unique
        }

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": dimension_grid,
                "y": 1 - dimension_grid,
                "z": grid_probability,
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
            }
        )

        for label in grid_label_not_nan_unique:

            z = grid_probability.copy()

            z[grid_label != label] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": dimension_grid,
                    "y": 1 - dimension_grid,
                    "z": z,
                    "opacity": opacity,
                    "colorscale": make_colorscale(
                        ("rgb(255, 255, 255)", label_color[label])
                    ),
                    "showscale": False,
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
                        "{}<br>{:.2e}".format(point, value)
                        for point, value in point_value.items()
                    ),
                    "marker": {
                        "opacity": point_value_opacity,
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
                    },
                },
            )
        )

    elif point_label is not None:

        for label in grid_label_not_nan_unique:

            is_label = point_label == label

            name = "Label {:.0f}".format(label)

            data.append(
                merge_2_dicts(
                    point_trace,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": point_x_dimension["x"][is_label],
                        "y": point_x_dimension["y"][is_label],
                        "text": point_x_dimension.index[is_label],
                        "marker": {"color": label_color[label]},
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
