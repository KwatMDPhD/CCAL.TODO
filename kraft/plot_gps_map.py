from numpy import isnan, nan, unique
from plotly.colors import make_colorscale

from .COLORBAR import COLORBAR
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .get_colorscale_color import get_colorscale_color
from .get_element_x_dimension_triangulation_edges import (
    get_element_x_dimension_triangulation_edges,
)
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly_figure import plot_plotly_figure


def plot_gps_map(
    node_x_dimension,
    element_x_dimension,
    element_label=None,
    dimension_grid=None,
    grid_probability=None,
    grid_label=None,
    label_colorscale=None,
    element_value=None,
    element_value_data_type="continuous",
    layout=None,
    element_trace=None,
    grid_label_opacity=0.8,
    elements_to_highlight=(),
    html_file_path=None,
):

    title_text = "{} {} + {} {}".format(
        node_x_dimension.index.size,
        node_x_dimension.index.name,
        element_x_dimension.index.size,
        element_x_dimension.index.name,
    )

    if element_value is not None:

        title_text += "<br>{}".format(element_value.name)

    axis = {"showgrid": False, "zeroline": False, "showticklabels": False}

    layout_template = {
        "height": 800,
        "width": 800,
        "title": {
            "x": 0.5,
            "text": "<b>{}</b>".format(title_text),
            "font": {"size": 24, "family": "gravitasone"},
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
                "size": 30,
                "color": "#23191e",
                "line": {"width": 2, "color": "#ffddca"},
            },
            "hoverinfo": "text",
        }
    )

    layout["annotations"] += [
        {
            "x": x,
            "y": y,
            "text": "<b>{}</b>".format(node),
            "showarrow": False,
            "font": {"size": 16, "family": "gravitasone", "color": "#20d9ba"},
            "yshift": 24,
        }
        for node, (x, y) in node_x_dimension.iterrows()
    ]

    element_trace_template = {
        "type": "scatter",
        "name": element_x_dimension.index.name,
        "mode": "markers",
        "marker": {
            "size": 16,
            "color": "#ffddca",
            "line": {"width": 1, "color": "#171412"},
            "opacity": 0.8,
        },
        "hoverinfo": "text",
    }

    if element_trace is None:

        element_trace = element_trace_template

    else:

        element_trace = merge_2_dicts_recursively(element_trace_template, element_trace)

    if grid_label is not None:

        grid_label_not_nan_unique = unique(grid_label[~isnan(grid_label)])

        label_color = {
            label: get_colorscale_color(
                label_colorscale, label, grid_label_not_nan_unique.size
            )
            for label in grid_label_not_nan_unique
        }

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": dimension_grid,
                "y": dimension_grid,
                "z": grid_probability[::-1],
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
                    "y": dimension_grid,
                    "z": z[::-1],
                    "opacity": grid_label_opacity,
                    "colorscale": make_colorscale(
                        ("rgb(255, 255, 255)", label_color[label])
                    ),
                    "showscale": False,
                    "hoverinfo": "none",
                }
            )

    if element_value is not None:

        element_value = element_value.reindex(index=element_x_dimension.index)

        element_value = element_value[element_value.abs().sort_values().index]

        element_x_dimension = element_x_dimension.loc[element_value.index]

        if element_value_data_type == "continuous":

            tickvals = element_value.describe()[
                ["min", "25%", "50%", "mean", "75%", "max"]
            ]

        else:

            tickvals = element_value.unique()

        data.append(
            merge_2_dicts_recursively(
                element_trace,
                {
                    "x": element_x_dimension["x"],
                    "y": element_x_dimension["y"],
                    "text": tuple(
                        "{} {:.2e}".format(element, value)
                        for element, value in element_value.items()
                    ),
                    "marker": {
                        "color": element_value,
                        "colorscale": DATA_TYPE_COLORSCALE[element_value_data_type],
                        "colorbar": merge_2_dicts_recursively(
                            COLORBAR,
                            {
                                "tickmode": "array",
                                "tickvals": tickvals,
                                "ticktext": tuple(
                                    "{:.2e}".format(tickval) for tickval in (tickvals)
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

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
