from matplotlib.colors import LinearSegmentedColormap, to_hex
from numpy import arange, asarray, cos, linspace, nan, pi, sin, unique
from pandas import Series

from .check_array_for_bad import check_array_for_bad
from .clip_array_by_standard_deviation import clip_array_by_standard_deviation
from .COLORBAR import COLORBAR
from .get_colormap_colors import get_colormap_colors
from .get_data_type import get_data_type
from .get_element_x_dimension_triangulation_edges import (
    get_element_x_dimension_triangulation_edges,
)
from .make_colorscale_from_colors import make_colorscale_from_colors
from .pick_colors import pick_colors
from .plot_plotly_figure import plot_plotly_figure

grid_extension = 1 / 1e3


def plot_gps_map(
    nodes,
    node_name,
    node_x_dimension,
    elements,
    element_name,
    element_x_dimension,
    element_marker_size,
    element_label,
    grid_values,
    grid_labels,
    label_colors,
    grid_label_opacity,
    annotation_x_element,
    annotation_types,
    annotation_std_maxs,
    annotation_colorscales,
    layout_size,
    highlight_binary,
    title_text,
    html_file_path,
):

    axis_template = {"showgrid": False, "zeroline": False, "showticklabels": False}

    layout = {
        "width": layout_size,
        "height": layout_size,
        "title": {"text": title_text, "font": {"color": "#4c221b"}},
        "xaxis": axis_template,
        "yaxis": axis_template,
    }

    data = []

    node_opacity = 0.88

    edge_xs, edge_ys = get_element_x_dimension_triangulation_edges(node_x_dimension)

    data.append(
        {
            "type": "scatter",
            "legendgroup": node_name,
            "showlegend": False,
            "x": edge_xs,
            "y": edge_ys,
            "line": {"color": "#23191e"},
            "opacity": node_opacity,
            "hoverinfo": None,
        }
    )

    node_color = "#2e211b"

    node_color_accent = "#ebf6f7"

    data.append(
        {
            "type": "scatter",
            "legendgroup": node_name,
            "name": node_name,
            "x": node_x_dimension[:, 0],
            "y": node_x_dimension[:, 1],
            "text": arange(len(nodes)),
            "mode": "markers+text",
            "marker": {
                "size": 32,
                "color": node_color,
                "line": {"width": 2.4, "color": node_color_accent},
            },
            "textfont": {"size": 16, "color": node_color_accent},
            "opacity": node_opacity,
        }
    )

    if grid_values is not None:

        x = linspace(0 - grid_extension, 1 + grid_extension, num=grid_values.shape[1])

        y = linspace(0 - grid_extension, 1 + grid_extension, num=grid_values.shape[0])

        data.append(
            {
                "type": "contour",
                "showlegend": False,
                "x": x,
                "y": y,
                "z": grid_values[::-1],
                "autocontour": False,
                "ncontours": 24,
                "contours": {"coloring": "none"},
                "showscale": False,
                "opacity": node_opacity,
            }
        )

        grid_labels_unique = unique(
            grid_labels[~check_array_for_bad(grid_labels, raise_for_bad=False)]
        ).astype(int)

        for label in grid_labels_unique:

            z = grid_values.copy()

            z[grid_labels != label] = nan

            data.append(
                {
                    "type": "heatmap",
                    "x": x,
                    "y": y,
                    "z": z[::-1],
                    "colorscale": make_colorscale_from_colors(
                        get_colormap_colors(
                            LinearSegmentedColormap.from_list(
                                None, ("#ffffff", label_colors[label])
                            )
                        )
                    ),
                    "showscale": False,
                    "opacity": grid_label_opacity,
                }
            )

    element_marker_line = {"width": element_marker_size / 10}

    element_opacity = 0.88

    if annotation_x_element is not None:

        if 1 < annotation_x_element.shape[0]:

            shapes = []

            marker_size_factor = element_marker_size / layout_size

        for i, (_, element_value) in enumerate(annotation_x_element.iterrows()):

            element_value.dropna(inplace=True)

            element_x_dimension_ = element_x_dimension[
                [elements.index(element) for element in element_value.index]
            ]

            if annotation_types is None:

                data_type = get_data_type(element_value)

            else:

                data_type = annotation_types[i]

            if data_type in ("binary", "categorical"):

                element_value = element_value.rank(method="dense") - 1

            elif data_type == "continuous":

                if annotation_std_maxs is not None:

                    element_value = Series(
                        clip_array_by_standard_deviation(
                            element_value.values,
                            annotation_std_maxs[i],
                            raise_for_bad=False,
                        ),
                        name=element_value.name,
                        index=element_value.index,
                    )

            if annotation_x_element.shape[0] == 1:

                sorted_indices = element_value.abs().argsort()

                element_value = element_value[sorted_indices]

                element_x_dimension_ = element_x_dimension_[sorted_indices]

            if annotation_colorscales is None:

                colorscale = make_colorscale_from_colors(
                    pick_colors(element_value, data_type=data_type)
                )

            else:

                colorscale = annotation_colorscales[i]

            if 1 == annotation_x_element.shape[0]:

                data.append(
                    {
                        "type": "scatter",
                        "showlegend": False,
                        "x": element_x_dimension_[:, 0],
                        "y": element_x_dimension_[:, 1],
                        "text": tuple(
                            "{}<br>{:.2f}".format(element, value)
                            for element, value in element_value.items()
                        ),
                        "mode": "markers",
                        "marker": {
                            "size": element_marker_size,
                            "color": element_value,
                            "colorscale": colorscale,
                            "showscale": data_type == "continuous",
                            "colorbar": COLORBAR,
                            "line": element_marker_line,
                        },
                        "opacity": element_opacity,
                        "hoverinfo": "text",
                    }
                )

                if highlight_binary and data_type == "binary":

                    layout["annotations"] = [
                        {
                            "x": element_x_dimension_[i, 0],
                            "y": element_x_dimension_[i, 1],
                            "text": "<b>{}</b>".format(element_value.index[i]),
                            "font": {"size": 16},
                            "arrowhead": 2,
                            "arrowwidth": 2,
                            "arrowcolor": "#c93756",
                            "standoff": 16,
                            "clicktoshow": "onoff",
                        }
                        for i in element_value.values.nonzero()[0]
                    ]

            else:

                for value, (x, y) in zip(element_value.values, element_x_dimension_):

                    color = to_hex(
                        colorscale[
                            int(
                                (len(colorscale) - 1)
                                * (value - element_value.min())
                                / (element_value.max() - element_value.min())
                            )
                        ][1]
                    )

                    sector_radian = pi * 2 / annotation_x_element.shape[0]

                    sector_radians = linspace(
                        sector_radian * i, sector_radian * (i + 1), num=16
                    )

                    path = "M {} {}".format(x, y)

                    for x_, y_ in zip(cos(sector_radians), sin(sector_radians)):

                        path += " L {} {}".format(
                            x + x_ * marker_size_factor, y + y_ * marker_size_factor
                        )

                    path += " Z"

                    shapes.append(
                        {
                            "type": "path",
                            "x0": x,
                            "y0": y,
                            "path": path,
                            "fillcolor": color,
                            "line": element_marker_line,
                            "opacity": element_opacity,
                        }
                    )

        if 1 < annotation_x_element.shape[0]:

            layout["shapes"] = shapes

    elif element_label is not None:

        for label in grid_labels_unique:

            element_indices = element_label == label

            label_str = str(label)

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
                        "color": label_colors[label],
                        "line": element_marker_line,
                    },
                    "opacity": element_opacity,
                    "hoverinfo": "text",
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
                    "color": "#ffffff",
                    "line": element_marker_line,
                },
                "opacity": element_opacity,
                "hoverinfo": "text",
            }
        )

    plot_plotly_figure({"layout": layout, "data": data}, html_file_path)
