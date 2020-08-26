from numpy import arange, argsort, meshgrid, nonzero, unique
from plotly.colors import (
    convert_colors_to_same_type,
    find_intermediate_color,
    make_colorscale,
    qualitative,
)
from plotly.io import show, write_html

from .array import normalize
from .dictionary import merge

COLORBAR = {
    "thicknessmode": "fraction",
    "thickness": 0.024,
    "len": 0.64,
    "ticks": "outside",
    "tickfont": {"size": 10},
}

DATA_TYPE_TO_COLORSCALE = {
    "continuous": make_colorscale(("#0000ff", "#ffffff", "#ff0000")),
    "categorical": make_colorscale(qualitative.Plotly),
    "binary": make_colorscale(("#006442", "#ffffff", "#ffa400")),
}


def plot_plotly(figure, file_path=None):

    figure = merge(figure, {"layout": {"autosize": False, "template": "plotly_white"}})

    config = {"editable": True}

    show(figure, config=config)

    if file_path is not None:

        write_html(figure, file_path, config=config)


def get_color(colorscale, number, maximum_number=None):

    if maximum_number is not None:

        if 1 <= number:

            number /= maximum_number - 1

    if colorscale is None:

        colorscale = make_colorscale(qualitative.Plotly)

    for index in range(len(colorscale) - 1):

        if colorscale[index][0] <= number <= colorscale[index + 1][0]:

            low_number, low_color = colorscale[index]

            high_number, high_color = colorscale[index + 1]

            color = find_intermediate_color(
                *convert_colors_to_same_type((low_color, high_color))[0],
                (number - low_number) / (high_number - low_number),
                colortype="rgb",
            )

            return "rgb{}".format(
                tuple(
                    int(float(intensity))
                    for intensity in color[4:-1].split(sep=",", maxsplit=2)
                )
            )


def plot_heat_map(
    matrix,
    axis_0_label_,
    axis_1_label_,
    axis_0_name,
    axis_1_name,
    colorscale=None,
    axis_0_group_=None,
    axis_1_group_=None,
    axis_0_group_colorscale=None,
    axis_1_group_colorscale=None,
    axis_0_group_to_name=None,
    axis_1_group_to_name=None,
    layout=None,
    axis_0_layout_annotation=None,
    axis_1_layout_annotation=None,
    file_path=None,
):

    if axis_0_group_ is not None:

        sort_index_ = argsort(axis_0_group_)

        axis_0_group_ = axis_0_group_[sort_index_]

        matrix = matrix[sort_index_]

        axis_0_label_ = axis_0_label_[sort_index_]

    if axis_1_group_ is not None:

        sort_index_ = argsort(axis_1_group_)

        axis_1_group_ = axis_1_group_[sort_index_]

        matrix = matrix[:, sort_index_]

        axis_1_label_ = axis_1_label_[sort_index_]

    domain = (0, 0.95)

    group_axis = {
        "domain": (0.96, 1),
        "showgrid": False,
        "showline": False,
        "zeroline": False,
        "showticklabels": False,
    }

    base = {
        "yaxis": {
            "title": "{} (n={})".format(axis_0_name, axis_0_label_.size),
            "domain": domain,
        },
        "xaxis": {
            "title": "{} (n={})".format(axis_1_name, axis_1_label_.size),
            "domain": domain,
        },
        "yaxis2": group_axis,
        "xaxis2": group_axis,
        "annotations": [],
    }

    if layout is None:

        layout = base

    else:

        layout = merge(base, layout)

    if colorscale is None:

        colorscale = DATA_TYPE_TO_COLORSCALE["continuous"]

    colorbar_x = 1.04

    data = [
        {
            "type": "heatmap",
            "z": matrix[::-1],
            "y": axis_0_label_[::-1],
            "x": axis_1_label_,
            "colorscale": colorscale,
            "colorbar": {**COLORBAR, "x": colorbar_x},
        }
    ]

    if axis_0_group_ is not None:

        axis_0_group_ = axis_0_group_[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": axis_0_group_.reshape((-1, 1)),
                "colorscale": axis_0_group_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+y",
            }
        )

        if axis_0_group_to_name is not None:

            base = {
                "xref": "x2",
                "x": 0,
                "xanchor": "left",
                "showarrow": False,
            }

            if axis_0_layout_annotation is not None:

                base = merge(base, axis_0_layout_annotation)

            for group in unique(axis_0_group_):

                index_0, index_1 = nonzero(axis_0_group_ == group)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "y": index_0 + (index_1 - index_0) / 2,
                        "text": axis_0_group_to_name[group],
                        **base,
                    }
                )

    if axis_1_group_ is not None:

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": axis_1_group_.reshape((1, -1)),
                "colorscale": axis_1_group_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+x",
            }
        )

        if axis_1_group_to_name is not None:

            base = {
                "yref": "y2",
                "y": 0,
                "yanchor": "bottom",
                "textangle": -90,
                "showarrow": False,
            }

            if axis_1_layout_annotation is not None:

                base = merge(base, axis_1_layout_annotation)

            for group in unique(axis_1_group_):

                index_0, index_1 = nonzero(axis_1_group_ == group)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "x": index_0 + (index_1 - index_0) / 2,
                        "text": axis_1_group_to_name[group],
                        **base,
                    }
                )

    plot_plotly({"data": data, "layout": layout}, file_path=file_path)


def plot_bubble_map(
    size_matrix,
    axis_0_label_,
    axis_1_label_,
    axis_0_name,
    axis_1_name,
    color_matrix=None,
    max_size=32,
    colorscale=None,
    layout=None,
    file_path=None,
):

    axis_0_size, axis_1_size = size_matrix.shape

    axis_0_grid = arange(axis_0_size)[::-1]

    axis_1_grid = arange(axis_1_size)

    base = {
        "height": max(480, axis_0_size * 2 * max_size),
        "width": max(480, axis_1_size * 2 * max_size),
        "yaxis": {
            "title": "{} (n={})".format(axis_0_name, axis_0_size),
            "tickvals": axis_0_grid,
            "ticktext": axis_0_label_,
        },
        "xaxis": {
            "title": "{} (n={})".format(axis_1_name, axis_1_size),
            "tickvals": axis_1_grid,
            "ticktext": axis_1_label_,
        },
    }

    if layout is None:

        layout = base

    else:

        layout = merge(base, layout)

    if color_matrix is None:

        color_matrix = size_matrix

    y_matrix, x_matrix = meshgrid(axis_0_grid, axis_1_grid)

    if colorscale is None:

        colorscale = DATA_TYPE_TO_COLORSCALE["continuous"]

    plot_plotly(
        {
            "data": [
                {
                    "y": y_matrix.ravel(),
                    "x": x_matrix.ravel(),
                    "text": size_matrix.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize(size_matrix, "0-1").ravel() * max_size,
                        "color": color_matrix.ravel(),
                        "colorscale": colorscale,
                        "colorbar": COLORBAR,
                    },
                }
            ],
            "layout": layout,
        },
        file_path=file_path,
    )


def plot_histogram(
    vector_,
    label__,
    name_,
    histnorm=None,
    bin_size=None,
    plot_rug=None,
    layout=None,
    file_path=None,
):

    if plot_rug is None:

        plot_rug = all(vector.size <= 1e3 for vector in vector_)

    data_number = len(name_)

    if plot_rug:

        height = 0.04

        yaxis_maximum = data_number * height

        yaxis2_minimum = yaxis_maximum + height

    else:

        yaxis_maximum = 0

        yaxis2_minimum = 0

    if histnorm is None:

        yaxis2_title_text = "Count"

    else:

        yaxis2_title_text = histnorm.title()

    base = {
        "xaxis": {"anchor": "y"},
        "yaxis": {
            "domain": (0, yaxis_maximum),
            "zeroline": False,
            "dtick": 1,
            "showticklabels": False,
        },
        "yaxis2": {"domain": (yaxis2_minimum, 1), "title": {"text": yaxis2_title_text}},
    }

    if layout is None:

        layout = base

    else:

        layout = merge(base, layout)

    data = []

    for data_index, (vector, label_, name), in enumerate(zip(vector_, label__, name_)):

        color = get_color(
            DATA_TYPE_TO_COLORSCALE["categorical"], data_index, data_number
        )

        base = {
            "legendgroup": data_index,
            "name": name,
            "x": vector,
        }

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "histnorm": histnorm,
                "xbins": {"size": bin_size},
                "marker": {"color": color},
                **base,
            }
        )

        if plot_rug:

            data.append(
                {
                    "showlegend": False,
                    "y": (data_index,) * vector.size,
                    "text": label_,
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                    "hoverinfo": "x+text",
                    **base,
                }
            )

    plot_plotly({"data": data, "layout": layout}, file_path=file_path)
