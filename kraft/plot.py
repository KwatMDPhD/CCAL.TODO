COLORBAR = {
    "thicknessmode": "fraction",
    "thickness": 0.024,
    "len": 0.64,
    "ticks": "outside",
    "tickfont": {"size": 10},
}
from numpy import arange, argsort, asarray, meshgrid, nonzero, unique
from pandas import DataFrame, Series
from plotly.colors import (
    convert_colors_to_same_type,
    find_intermediate_color,
    make_colorscale,
    qualitative,
)
from plotly.io import show, templates, write_html

from .array import normalize
from .support import cast_builtin, merge_2_dicts

DATA_TYPE_COLORSCALE = {
    "continuous": make_colorscale(("#0000ff", "#ffffff", "#ff0000")),
    "categorical": make_colorscale(qualitative.Plotly),
    "binary": make_colorscale(("#ffddca", "#006442")),
}


def get_color(colorscale, value, n=None):

    if 1 < value or (value == 1 and n is not None):

        n_block = n - 1

        if value == n_block:

            value = 1

        else:

            value = value / n_block

    if colorscale is None:

        colorscale = make_colorscale(qualitative.Plotly)

    for i in range(len(colorscale) - 1):

        if colorscale[i][0] <= value <= colorscale[i + 1][0]:

            scale_low, color_low = colorscale[i]

            scale_high, color_high = colorscale[i + 1]

            value_ = (value - scale_low) / (scale_high - scale_low)

            color = find_intermediate_color(
                *convert_colors_to_same_type((color_low, color_high))[0],
                value_,
                colortype="rgb",
            )

            return "rgb{}".format(
                tuple(int(float(i)) for i in color[4:-1].split(sep=","))
            )


def plot_heat_map(
    matrix,
    colorscale=None,
    ordered_annotation=False,
    row_annotations=None,
    row_annotation_colorscale=None,
    row_annotation_str=None,
    column_annotations=None,
    column_annotation_colorscale=None,
    column_annotation_str=None,
    layout=None,
    layout_annotation_row=None,
    layout_annotation_column=None,
    html_file_path=None,
):

    if not isinstance(matrix, DataFrame):

        matrix = DataFrame(matrix)

    if row_annotations is not None:

        row_annotations = asarray(row_annotations)

        if not ordered_annotation:

            index = argsort(row_annotations)

            row_annotations = row_annotations[index]

            matrix = matrix.iloc[index]

    if column_annotations is not None:

        column_annotations = asarray(column_annotations)

        if not ordered_annotation:

            index = argsort(column_annotations)

            column_annotations = column_annotations[index]

            matrix = matrix.iloc[:, index]

    heat_map_axis_ = {"domain": (0, 0.95)}

    annotation_axis_ = {"domain": (0.96, 1), "showticklabels": False}

    layout_ = {
        "xaxis": {
            "title": "{} (n={})".format(matrix.columns.name, matrix.columns.size),
            **heat_map_axis_,
        },
        "yaxis": {
            "title": "{} (n={})".format(matrix.index.name, matrix.index.size),
            **heat_map_axis_,
        },
        "xaxis2": annotation_axis_,
        "yaxis2": annotation_axis_,
        "annotations": [],
    }

    if layout is None:

        layout = layout_

    else:

        layout = merge_2_dicts(layout_, layout)

    if any(isinstance(cast_builtin(i), str) for i in matrix.columns):

        x = matrix.columns

    else:

        x = None

    if any(isinstance(cast_builtin(i), str) for i in matrix.index):

        y = matrix.index[::-1]

    else:

        y = None

    if colorscale is None:

        colorscale = DATA_TYPE_COLORSCALE["continuous"]

    colorbar_x = 1.05

    data = [
        {
            "type": "heatmap",
            "x": x,
            "y": y,
            "z": matrix.values[::-1],
            "colorscale": colorscale,
            "colorbar": {**COLORBAR, "x": colorbar_x},
        }
    ]

    annotation_ = {"showarrow": False}

    if row_annotations is not None:

        row_annotations = row_annotations[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": row_annotations.reshape(row_annotations.size, 1),
                "colorscale": row_annotation_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+y",
            }
        )

        if row_annotation_str is not None:

            layout_annotation_row_ = {
                "xref": "x2",
                "x": 0,
                "xanchor": "left",
                **annotation_,
            }

            if layout_annotation_row is None:

                layout_annotation_row = layout_annotation_row_

            else:

                layout_annotation_row = merge_2_dicts(
                    layout_annotation_row_, layout_annotation_row
                )

            for i in unique(row_annotations):

                index_0, index__1 = nonzero(row_annotations == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "y": index_0 + (index__1 - index_0) / 2,
                        "text": row_annotation_str[i],
                        **layout_annotation_row,
                    }
                )

    if column_annotations is not None:

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": column_annotations.reshape(column_annotations.size, 1),
                "transpose": True,
                "colorscale": column_annotation_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+x",
            }
        )

        if column_annotation_str is not None:

            layout_column_annotation_ = {
                "yref": "y2",
                "y": 0,
                "yanchor": "bottom",
                "textangle": -90,
                **annotation_,
            }

            if layout_annotation_column is None:

                layout_annotation_column = layout_column_annotation_

            else:

                layout_annotation_column = merge_2_dicts(
                    layout_column_annotation_, layout_annotation_column
                )

            for i in unique(column_annotations):

                index_0, index__1 = nonzero(column_annotations == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "x": index_0 + (index__1 - index_0) / 2,
                        "text": column_annotation_str[i],
                        **layout_annotation_column,
                    }
                )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)


templates["kraft"] = {"layout": {"autosize": False}}


def plot_plotly(figure, html_file_path=None):

    template = "plotly_white+kraft"

    if "layout" in figure:

        figure["layout"]["template"] = template

    else:

        figure["layout"] = {"template": template}

    config = {"editable": True}

    show(figure, config=config)

    if html_file_path is not None:

        write_html(figure, html_file_path, config=config)


def plot_bubble_map(
    dataframe_size,
    dataframe_color=None,
    max_size=20,
    colorscale=None,
    layout=None,
    html_file_path=None,
):

    x_grid = arange(dataframe_size.shape[1])

    y_grid = arange(dataframe_size.shape[0])[::-1]

    layout_ = {
        "height": max(500, max_size * 2 * dataframe_size.shape[0]),
        "width": max(500, max_size * 2 * dataframe_size.shape[1]),
        "xaxis": {
            "title": "{} (n={})".format(
                dataframe_size.columns.name, dataframe_size.columns.size
            ),
            "tickvals": x_grid,
            "ticktext": dataframe_size.columns,
        },
        "yaxis": {
            "title": "{} (n={})".format(
                dataframe_size.index.name, dataframe_size.index.size
            ),
            "tickvals": y_grid,
            "ticktext": dataframe_size.index,
        },
    }

    if layout is None:

        layout = layout_

    else:

        layout = merge_2_dicts(layout_, layout)

    if dataframe_color is None:

        dataframe_color = dataframe_size

    mesh_grid_x, mesh_grid_y = meshgrid(x_grid, y_grid)

    if colorscale is None:

        colorscale = DATA_TYPE_COLORSCALE["continuous"]

    plot_plotly(
        {
            "layout": layout,
            "data": [
                {
                    "x": mesh_grid_x.ravel(),
                    "y": mesh_grid_y.ravel(),
                    "text": dataframe_size.values.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize(dataframe_size.values, "0-1").ravel()
                        * max_size,
                        "color": dataframe_color.values.ravel(),
                        "colorscale": colorscale,
                        "colorbar": COLORBAR,
                    },
                }
            ],
        },
        html_file_path=html_file_path,
    )


def plot_histogram(
    xs, histnorm=None, bin_size=None, plot_rug=None, layout=None, html_file_path=None
):

    if histnorm is None:

        yaxis2_title_text = "N"

    else:

        yaxis2_title_text = histnorm.title()

    if plot_rug is None:

        plot_rug = all(len(x) < 1e3 for x in xs)

    n_x = len(xs)

    if plot_rug:

        rug_height = 0.04

        yaxis_domain_max = n_x * rug_height

        yaxis2_domain_min = yaxis_domain_max + rug_height

    else:

        yaxis_domain_max = 0

        yaxis2_domain_min = 0

    yaxis_domain = 0, yaxis_domain_max

    yaxis2_domain = yaxis2_domain_min, 1

    layout_ = {
        "xaxis": {"anchor": "y"},
        "yaxis": {
            "domain": yaxis_domain,
            "zeroline": False,
            "dtick": 1,
            "showticklabels": False,
        },
        "yaxis2": {"domain": yaxis2_domain, "title": {"text": yaxis2_title_text}},
        "showlegend": True,
    }

    if layout is None:

        layout = layout_

    else:

        layout = merge_2_dicts(layout_, layout)

    data = []

    for x_index, x in enumerate(xs):

        if isinstance(x, Series):

            name = x.name

            text = x.index

        else:

            name = x_index

            text = None

        color = get_color(DATA_TYPE_COLORSCALE["categorical"], x_index, n_x)

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "legendgroup": x_index,
                "name": name,
                "showlegend": 1 < len(xs),
                "x": x,
                "histnorm": histnorm,
                "xbins": {"size": bin_size},
                "marker": {"color": color},
            }
        )

        if plot_rug:

            data.append(
                {
                    "legendgroup": x_index,
                    "name": name,
                    "showlegend": False,
                    "x": x,
                    "y": (x_index,) * len(x),
                    "text": text,
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                    "hoverinfo": "x+text",
                }
            )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
