from numpy import asarray, isnan, nan, unique
from pandas import Series
from plotly.colors import make_colorscale

from .clip_array_by_standard_deviation import clip_array_by_standard_deviation
from .COLORBAR import COLORBAR
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .get_colorscale_color import get_colorscale_color
from .get_element_x_dimension_triangulation_edges import (
    get_element_x_dimension_triangulation_edges,
)
from .merge_2_dicts_recursively import merge_2_dicts_recursively
from .plot_plotly_figure import plot_plotly_figure


def plot_gps_map(
    nodes,
    node_name,
    node_x_dimension,
    elements,
    element_name,
    element_x_dimension,
    dimension_grid=None,
    element_labels=None,
    grid_values=None,
    grid_labels=None,
    label_colorscale=DATA_TYPE_COLORSCALE["categorical"],
    element_value=None,
    element_value_data_type="continuous",
    element_value_std_max=None,
    layout=None,
    element_value_binary_annotation=False,
    html_file_path=None,
    node_marker_size=30,
    node_marker_color="#2e211b",
    node_line_color="#23191e",
    node_textfont_size=15,
    node_textfont_color="#ffffff",
    element_marker_size=20,
    element_marker_color="#ebf6f7",
    element_marker_line_width=1,
    element_marker_line_color="#2e211b",
    n_contour=20,
    grid_label_opacity=0.8,
):
    axis = {"showgrid": False, "zeroline": False, "showticklabels": False}
    layout_template = {
        "height": 800,
        "width": 800,
        "title": {
            "xref": "paper",
            "x": 0.5,
            "text": "<b>{} {}<br>{} {}</b>".format(
                len(nodes), node_name, len(elements), element_name
            ),
        },
        "xaxis": axis,
        "yaxis": axis,
    }
    if layout is None:
        layout = layout_template
    else:
        layout = merge_2_dicts_recursively(layout_template, layout)

    edge_xs, edge_ys = get_element_x_dimension_triangulation_edges(node_x_dimension)

    data = [
        {
            "type": "scatter",
            "legendgroup": node_name,
            "showlegend": False,
            "x": edge_xs,
            "y": edge_ys,
            "line": {"color": node_line_color},
        },
        {
            "type": "scatter",
            "legendgroup": node_name,
            "name": node_name,
            "x": node_x_dimension[:, 0],
            "y": node_x_dimension[:, 1],
            "text": nodes,
            "mode": "markers+text",
            "marker": {"size": node_marker_size, "color": node_marker_color},
            "textfont": {"size": node_textfont_size, "color": node_textfont_color},
            "hoverinfo": "text",
        },
    ]

    element_template = {
        "type": "scatter",
        "name": element_name,
        "mode": "markers",
        "marker": {
            "size": element_marker_size,
            "color": element_marker_color,
            "line": {
                "width": element_marker_line_width,
                "color": element_marker_line_color,
            },
        },
        "hoverinfo": "text",
    }

    if grid_labels is not None:

        grid_labels_not_nan_unique = unique(grid_labels[~isnan(grid_labels)])

        label_color = {
            label: get_colorscale_color(
                label_colorscale, label / (grid_labels_not_nan_unique.size - 1)
            )
            for label in grid_labels_not_nan_unique
        }

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": dimension_grid,
                "y": dimension_grid,
                "z": grid_values[::-1],
                "autocontour": False,
                "ncontours": n_contour,
                "contours": {"coloring": "none"},
            }
        )

        for label in grid_labels_not_nan_unique:
            z = grid_values.copy()
            z[grid_labels != label] = nan
            data.append(
                {
                    "type": "heatmap",
                    "x": dimension_grid,
                    "y": dimension_grid,
                    "z": z[::-1],
                    "colorscale": make_colorscale(
                        ("rgb(255, 255, 255)", label_color[label])
                    ),
                    "opacity": grid_label_opacity,
                    "showscale": False,
                    "hoverinfo": "none",
                }
            )

    if element_value is not None:

        element_value = element_value[element_value.abs().sort_values().index]

        element_x_dimension_ = element_x_dimension[
            [elements.index(element) for element in element_value.index]
        ]

        if (
            element_value_data_type == "continuous"
            and element_value_std_max is not None
        ):

            element_value = Series(
                clip_array_by_standard_deviation(
                    element_value.values, element_value_std_max, raise_for_bad=False
                ),
                index=element_value.index,
                name=element_value.name,
            )

        tickvals = element_value.describe()[["min", "25%", "50%", "mean", "75%", "max"]]
        data.append(
            merge_2_dicts_recursively(
                element_template,
                {
                    "x": element_x_dimension_[:, 0],
                    "y": element_x_dimension_[:, 1],
                    "text": tuple(
                        "{}<br>{:.2f}".format(element, value)
                        for element, value in element_value.items()
                    ),
                    "marker": {
                        "color": element_value,
                        "colorscale": DATA_TYPE_COLORSCALE[element_value_data_type],
                        "colorbar": merge_2_dicts_recursively(
                            COLORBAR,
                            {
                                "title": {"text": element_value.name},
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

        if (
            element_value_data_type == "binary"
            and element_value_binary_annotation is not None
        ):

            layout["annotations"] = [
                {
                    "x": element_x_dimension_[i, 0],
                    "y": element_x_dimension_[i, 1],
                    "text": "<b>{}</b>".format(element_value.index[i]),
                    "arrowhead": 2,
                    "arrowwidth": 2,
                    "arrowcolor": "#c93756",
                    "standoff": None,
                    "clicktoshow": "onoff",
                    **element_value_binary_annotation,
                }
                for i in element_value.values.nonzero()[0]
            ]

    elif element_labels is not None:
        for label in grid_labels_not_nan_unique:
            is_label = element_labels == label
            data.append(
                merge_2_dicts_recursively(
                    element_template,
                    {
                        "name": "{:.0f}".format(label),
                        "x": element_x_dimension[is_label, 0],
                        "y": element_x_dimension[is_label, 1],
                        "text": asarray(elements)[is_label],
                        "marker": {"color": label_color[label]},
                    },
                )
            )

    else:
        data.append(
            merge_2_dicts_recursively(
                element_template,
                {
                    "x": element_x_dimension[:, 0],
                    "y": element_x_dimension[:, 1],
                    "text": elements,
                },
            )
        )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
