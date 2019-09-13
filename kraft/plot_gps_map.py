from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .normalize_series import normalize_series
from .get_colorscale_color import get_colorscale_color
from plotly.colors import make_colorscale
from .merge_2_dicts_recursively import merge_2_dicts_recursively

from numpy import arange, asarray, cos, linspace, nan, pi, sin, unique, isnan
from pandas import Series

from .clip_array_by_standard_deviation import clip_array_by_standard_deviation
from .COLORBAR import COLORBAR
from .get_element_x_dimension_triangulation_edges import (
    get_element_x_dimension_triangulation_edges,
)
from .plot_plotly_figure import plot_plotly_figure


def plot_gps_map(
    nodes,
    node_name,
    node_x_dimension,
    elements,
    element_name,
    element_x_dimension,
    dimension_grid=None,
    element_label=None,
    grid_values=None,
    grid_labels=None,
    label_colorscale=DATA_TYPE_COLORSCALE["categorical"],
    element_value=None,
    element_value_data_type="continuous",
    element_value_std_max=nan,
    highlight_binary=False,
    layout=None,
    html_file_path=None,
    node_marker_size=30,
    node_marker_color="#2e211b",
    node_line_color="#23191e",
    node_textfont_size=20,
    node_textfont_color="#ffffff",
    element_marker_size=20,
    element_marker_color="#ebf6f7",
    element_marker_line_width=1,
    element_marker_line_color="#2e211b",
):
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
        "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
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
            "text": arange(len(nodes)),
            "mode": "markers+text",
            "marker": {"size": node_marker_size, "color": node_marker_color},
            "textfont": {"size": node_textfont_size, "color": node_textfont_color},
            "hoverinfo": "text",
        },
    ]

    if element_label is not None:
        label_color = {
            label: get_colorscale_color(
                label_colorscale, label / (len(set(element_label)) - 1)
            )
            for label in set(element_label)
        }
        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": dimension_grid,
                "y": dimension_grid,
                "z": grid_values[::-1],
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
                "showscale": False,
            }
        )
        grid_labels_not_nan_unique = unique(grid_labels[~isnan(grid_labels)])
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
                }
            )

    if element_value is not None:

        pass

        # if 1 < annotation_x_element.shape[0]:
        #     shapes = []
        #     marker_size_factor = 1

        # for i, (_, element_value) in enumerate(annotation_x_element.iterrows()):

        #     element_value.dropna(inplace=True)

        #     element_x_dimension_ = element_x_dimension[
        #         [elements.index(element) for element in element_value.index]
        #     ]

        #     data_type = annotation_types[i]

        #     if data_type == "continuous" and annotation_std_maxs is not None:

        #         element_value = Series(
        #             clip_array_by_standard_deviation(
        #                 element_value.values,
        #                 annotation_std_maxs[i],
        #                 raise_for_bad=False,
        #             ),
        #             name=element_value.name,
        #             index=element_value.index,
        #         )

        #     elif data_type in ("categorical", "binary"):

        #         if (element_value < 0).any():

        #             raise

        #     if annotation_x_element.shape[0] == 1:
        #         sorted_indices = element_value.abs().argsort()
        #         element_value = element_value[sorted_indices]
        #         element_x_dimension_ = element_x_dimension_[sorted_indices]

        #     if annotation_colorscales is None:
        #         colorscale = DATA_TYPE_COLORSCALE[data_type]
        #     else:
        #         colorscale = annotation_colorscales[i]

        #     if 1 == annotation_x_element.shape[0]:

        #         data.append(
        #             {
        #                 "type": "scatter",
        #                 "showlegend": False,
        #                 "x": element_x_dimension_[:, 0],
        #                 "y": element_x_dimension_[:, 1],
        #                 "text": tuple(
        #                     "{}<br>{:.2f}".format(element, value)
        #                     for element, value in element_value.items()
        #                 ),
        #                 "mode": "markers",
        #                 "marker": {
        #                     "size": element_marker_size,
        #                     "color": element_value,
        #                     "colorscale": colorscale,
        #                     "showscale": data_type == "continuous",
        #                     "colorbar": COLORBAR,
        #                     "line": element_marker_line,
        #                 },
        #                 "opacity": element_opacity,
        #                 "hoverinfo": "text",
        #             }
        #         )

        #         if highlight_binary and data_type == "binary":
        #             layout["annotations"] = [
        #                 {
        #                     "x": element_x_dimension_[i, 0],
        #                     "y": element_x_dimension_[i, 1],
        #                     "text": "<b>{}</b>".format(element_value.index[i]),
        #                     "font": {"size": None},
        #                     "arrowhead": 2,
        #                     "arrowwidth": 2,
        #                     "arrowcolor": "#c93756",
        #                     "standoff": None,
        #                     "clicktoshow": "onoff",
        #                 }
        #                 for i in element_value.values.nonzero()[0]
        #             ]

        #     else:
        #         if data_type == "continuous":
        #             element_value = normalize_series(element_value, "0-1")
        #         else:
        #             element_value /= element_value.max() - 1

        #         for value, (x, y) in zip(element_value.values, element_x_dimension_):

        #             sector_radian = pi * 2 / annotation_x_element.shape[0]
        #             sector_radians = linspace(
        #                 sector_radian * i, sector_radian * (i + 1), num=16
        #             )
        #             path = "M {} {}".format(x, y)
        #             for x_, y_ in zip(cos(sector_radians), sin(sector_radians)):
        #                 path += " L {} {}".format(
        #                     x + x_ * marker_size_factor, y + y_ * marker_size_factor
        #                 )
        #             path += " Z"
        #             shapes.append(
        #                 {
        #                     "type": "path",
        #                     "x0": x,
        #                     "y0": y,
        #                     "path": path,
        #                     "fillcolor": get_colorscale_color(colorscale, value),
        #                     "line": element_marker_line,
        #                     "opacity": element_opacity,
        #                 }
        #             )

        # if 1 < annotation_x_element.shape[0]:

        #     layout["shapes"] = shapes

    elif element_label is not None:
        for label in grid_labels_not_nan_unique:
            element_indices = element_label == label
            label_str = "{}".format(label)
            data.append(
                {
                    "type": "scatter",
                    "legendgroup": label_str,
                    "name": label_str,
                    "x": element_x_dimension[element_indices, 0],
                    "y": element_x_dimension[element_indices, 1],
                    "text": asarray(elements)[element_indices],
                    "mode": "markers",
                    "marker": {
                        "size": element_marker_size,
                        "color": label_color[label],
                    },
                }
            )
    else:
        data.append(
            {
                "type": "scatter",
                "name": element_name,
                "x": element_x_dimension[:, 0],
                "y": element_x_dimension[:, 1],
                "text": elements,
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
        )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
