from numpy import isnan, nan, unique
from pandas import DataFrame
from plotly.colors import make_colorscale

from .COLORBAR import COLORBAR
from .get_colorscale_color import get_colorscale_color
from .get_element_x_dimension_triangulation_edges import (
    get_element_x_dimension_triangulation_edges,
)
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly import plot_plotly


def plot_gps_map(
    node_x_dimension,
    element_x_dimension,
    node_marker_size=16,
    opacity=0.8,
    element_label=None,
    dimension_grid=None,
    grid_probability=None,
    grid_label=None,
    element_label_colorscale=None,
    element_value=None,
    element_value_na_opacity=None,
    element_value_colorscale=None,
    ticktext_function=None,
    layout=None,
    show_node_text=True,
    element_trace=None,
    elements_to_highlight=(),
    html_file_path=None,
):

    node_x_dimension = DataFrame(
        {
            "x": node_x_dimension.iloc[:, 1].values,
            "y": 1 - node_x_dimension.iloc[::1, 0].values,
        },
        index=node_x_dimension.index,
    )

    element_x_dimension = DataFrame(
        {
            "x": element_x_dimension.iloc[:, 1].values,
            "y": 1 - element_x_dimension.iloc[::1, 0].values,
        },
        index=element_x_dimension.index,
    )

    title_text = "{} {} & {} {}".format(
        node_x_dimension.index.size,
        node_x_dimension.index.name,
        element_x_dimension.index.size,
        element_x_dimension.index.name,
    )

    if element_value is not None:

        title_text = "{}<br>{}".format(title_text, element_value.name)

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

        layout = merge_2_dicts_recursively(layout_template, layout)

    edge_xs, edge_ys = get_element_x_dimension_triangulation_edges(node_x_dimension)

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

    element_trace_template = {
        "type": "scatter",
        "name": element_x_dimension.index.name,
        "mode": "markers",
        "marker": {
            "size": 16,
            "color": "#20d9ba",
            "line": {"width": 0.8, "color": "#ebf6f7"},
            "opacity": opacity,
        },
        "hoverinfo": "text",
    }

    if element_trace is None:

        element_trace = element_trace_template

    else:

        element_trace = merge_2_dicts_recursively(element_trace_template, element_trace)

    if grid_label is not None:

        grid_label_not_nan_unique = unique(grid_label[~isnan(grid_label)])

        n_unique_label = grid_label_not_nan_unique.size

        label_color = {
            label: get_colorscale_color(element_label_colorscale, label, n_unique_label)
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

    if element_value is not None:

        element_value = element_value.reindex(index=element_x_dimension.index)

        element_value_opacity = opacity

        if element_value_na_opacity is None:

            element_value.dropna(inplace=True)

        else:

            element_value_opacity = element_value.where(
                isnan, other=element_value_opacity
            ).fillna(value=element_value_na_opacity)

        element_value = element_value[
            element_value.abs().sort_values(na_position="first").index
        ]

        element_x_dimension = element_x_dimension.loc[element_value.index]

        if element_value.astype(float).map(float.is_integer).all():

            tickvals = element_value.unique()

            if ticktext_function is None:

                ticktext_function = "{:.0f}".format

        else:

            tickvals = element_value.describe()[
                ["min", "25%", "50%", "mean", "75%", "max"]
            ].values

            ticktext_function = "{:.2e}".format

        data.append(
            merge_2_dicts_recursively(
                element_trace,
                {
                    "x": element_x_dimension["x"],
                    "y": element_x_dimension["y"],
                    "text": tuple(
                        "{}<br>{:.2e}".format(element, value)
                        for element, value in element_value.items()
                    ),
                    "marker": {
                        "opacity": element_value_opacity,
                        "color": element_value,
                        "colorscale": element_value_colorscale,
                        "colorbar": merge_2_dicts_recursively(
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

    elif element_label is not None:

        for label in grid_label_not_nan_unique:

            is_label = element_label == label

            name = "Label {:.0f}".format(label)

            data.append(
                merge_2_dicts_recursively(
                    element_trace,
                    {
                        "legendgroup": name,
                        "name": name,
                        "x": element_x_dimension["x"][is_label],
                        "y": element_x_dimension["y"][is_label],
                        "text": element_x_dimension.index[is_label],
                        "marker": {"color": label_color[label]},
                    },
                )
            )

    else:

        data.append(
            merge_2_dicts_recursively(
                element_trace,
                {
                    "x": element_x_dimension["x"],
                    "y": element_x_dimension["y"],
                    "text": element_x_dimension.index,
                },
            )
        )

    layout["annotations"] += [
        {
            "x": element_x_dimension.loc[element, "x"],
            "y": element_x_dimension.loc[element, "y"],
            "text": "<b>{}</b>".format(element),
            "arrowhead": 2,
            "arrowwidth": 2,
            "arrowcolor": "#c93756",
            "standoff": None,
            "clicktoshow": "onoff",
        }
        for element in elements_to_highlight
    ]

    plot_plotly({"layout": layout, "data": data}, html_file_path)
