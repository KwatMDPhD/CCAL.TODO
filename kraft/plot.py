from numpy import arange, argsort, meshgrid, nonzero, unique
from plotly.colors import (
    convert_colors_to_same_type,
    find_intermediate_color,
    make_colorscale,
    qualitative,
)
from plotly.io import show, templates, write_html

from .array import normalize
from .dict_ import merge
from .support import cast_builtin

templates["kraft"] = {"layout": {"autosize": False}}

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
    "binary": make_colorscale(("#ffddca", "#006442")),
}


def plot_plotly(figure, html_file_path=None):

    template = "plotly_white+kraft"

    if "layout" in figure:

        figure["layout"]["template"] = template

    else:

        figure["layout"] = {"template": template}

    config = {"editable": True}

    show(figure, config=config)

    if html_file_path is not None:

        assert html_file_path.endswith(".html")

        write_html(figure, html_file_path, config=config)


def get_color(colorscale, value, n=None):

    if n is not None:

        assert isinstance(cast_builtin(n), int)

        assert isinstance(cast_builtin(value), int)

        if 1 <= value < n:

            n_block = n - 1

            if value == n_block:

                value = 1.0

            else:

                value = value / n_block

    if colorscale is None:

        colorscale = make_colorscale(qualitative.Plotly)

    for i in range(len(colorscale) - 1):

        if colorscale[i][0] <= value <= colorscale[i + 1][0]:

            scale_low, color_low = colorscale[i]

            scale_high, color_high = colorscale[i + 1]

            color = find_intermediate_color(
                *convert_colors_to_same_type((color_low, color_high))[0],
                (value - scale_low) / (scale_high - scale_low),
                colortype="rgb",
            )

            return "rgb{}".format(
                tuple(int(float(i)) for i in color[4:-1].split(sep=","))
            )


def plot_heat_map(
    dataframe,
    colorscale=None,
    sort_groups=True,
    axis_0_groups=None,
    axis_0_group_colorscale=None,
    axis_0_group_to_name=None,
    axis_1_groups=None,
    axis_1_group_colorscale=None,
    axis_1_group_to_name=None,
    layout=None,
    layout_annotation_axis_0=None,
    layout_annotation_axis_1=None,
    html_file_path=None,
):

    if axis_0_groups is not None:

        if sort_groups:

            is_ = argsort(axis_0_groups)

            axis_0_groups = axis_0_groups[is_]

            dataframe = dataframe.iloc[is_, :]

    if axis_1_groups is not None:

        if sort_groups:

            is_ = argsort(axis_1_groups)

            axis_1_groups = axis_1_groups[is_]

            dataframe = dataframe.iloc[:, is_]

    heat_map_axis_template = {"domain": (0, 0.95)}

    group_axis = {"domain": (0.96, 1), "showticklabels": False}

    axis_0_is_str = any(
        isinstance(cast_builtin(label), str) for label in dataframe.index
    )

    axis_1_is_str = any(
        isinstance(cast_builtin(label), str) for label in dataframe.columns
    )

    layout_template = {
        "xaxis": {
            "showticklabels": axis_1_is_str,
            "title": "{} (n={})".format(dataframe.columns.name, dataframe.columns.size),
            **heat_map_axis_template,
        },
        "yaxis": {
            "showticklabels": axis_0_is_str,
            "title": "{} (n={})".format(dataframe.index.name, dataframe.index.size),
            **heat_map_axis_template,
        },
        "xaxis2": group_axis,
        "yaxis2": group_axis,
        "annotations": [],
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge(layout_template, layout)

    if axis_1_is_str:

        x = dataframe.columns.to_numpy()

    else:

        x = None

    if axis_0_is_str:

        y = dataframe.index.to_numpy()[::-1]

    else:

        y = None

    if colorscale is None:

        colorscale = DATA_TYPE_TO_COLORSCALE["continuous"]

    colorbar_x = 1.05

    data = [
        {
            "type": "heatmap",
            "x": x,
            "y": y,
            "z": dataframe.to_numpy()[::-1],
            "colorscale": colorscale,
            "colorbar": {**COLORBAR, "x": colorbar_x},
        }
    ]

    annotation_template = {"showarrow": False}

    if axis_0_groups is not None:

        axis_0_groups = axis_0_groups[::-1]

        colorbar_x += 0.1

        data.append(
            {
                "xaxis": "x2",
                "type": "heatmap",
                "z": axis_0_groups.reshape(axis_0_groups.size, 1),
                "colorscale": axis_0_group_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+y",
            }
        )

        if axis_0_group_to_name is not None:

            layout_annotation_axis_0_template = {
                "xref": "x2",
                "x": 0,
                "xanchor": "left",
                **annotation_template,
            }

            if layout_annotation_axis_0 is None:

                layout_annotation_axis_0 = layout_annotation_axis_0_template

            else:

                layout_annotation_axis_0 = merge(
                    layout_annotation_axis_0_template, layout_annotation_axis_0
                )

            for i in unique(axis_0_groups):

                i_0, i_1 = nonzero(axis_0_groups == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "y": i_0 + (i_1 - i_0) / 2,
                        "text": axis_0_group_to_name[i],
                        **layout_annotation_axis_0,
                    }
                )

    if axis_1_groups is not None:

        colorbar_x += 0.1

        data.append(
            {
                "yaxis": "y2",
                "type": "heatmap",
                "z": axis_1_groups.reshape(axis_1_groups.size, 1),
                "transpose": True,
                "colorscale": axis_1_group_colorscale,
                "colorbar": {**COLORBAR, "x": colorbar_x, "dtick": 1},
                "hoverinfo": "z+x",
            }
        )

        if axis_1_group_to_name is not None:

            layout_annotation_axis_1_template = {
                "yref": "y2",
                "y": 0,
                "yanchor": "bottom",
                "textangle": -90,
                **annotation_template,
            }

            if layout_annotation_axis_1 is None:

                layout_annotation_axis_1 = layout_annotation_axis_1_template

            else:

                layout_annotation_axis_1 = merge(
                    layout_annotation_axis_1_template, layout_annotation_axis_1
                )

            for i in unique(axis_1_groups):

                i_0, i_1 = nonzero(axis_1_groups == i)[0][[0, -1]]

                layout["annotations"].append(
                    {
                        "x": i_0 + (i_1 - i_0) / 2,
                        "text": axis_1_group_to_name[i],
                        **layout_annotation_axis_1,
                    }
                )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)


def plot_bubble_map(
    dataframe_size,
    dataframe_color=None,
    max_size=32,
    colorscale=None,
    layout=None,
    html_file_path=None,
):

    x_grid = arange(dataframe_size.shape[1])

    y_grid = arange(dataframe_size.shape[0])[::-1]

    layout_template = {
        "height": max(480, max_size * 2 * dataframe_size.shape[0]),
        "width": max(480, max_size * 2 * dataframe_size.shape[1]),
        "xaxis": {
            "title": "{} (n={})".format(
                dataframe_size.columns.name, dataframe_size.columns.size
            ),
            "tickvals": x_grid,
            "ticktext": dataframe_size.columns.to_numpy(),
        },
        "yaxis": {
            "title": "{} (n={})".format(
                dataframe_size.index.name, dataframe_size.index.size
            ),
            "tickvals": y_grid,
            "ticktext": dataframe_size.index.to_numpy(),
        },
    }

    if layout is None:

        layout = layout_template

    else:

        layout = merge(layout_template, layout)

    if dataframe_color is None:

        dataframe_color = dataframe_size

    x, y = meshgrid(x_grid, y_grid)

    if colorscale is None:

        colorscale = DATA_TYPE_TO_COLORSCALE["continuous"]

    matrix_size = dataframe_size.to_numpy()

    plot_plotly(
        {
            "layout": layout,
            "data": [
                {
                    "x": x.ravel(),
                    "y": y.ravel(),
                    "text": matrix_size.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize(matrix_size, "0-1").ravel() * max_size,
                        "color": dataframe_color.to_numpy().ravel(),
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

        plot_rug = all(x.size < 1e3 for x in xs)

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

    layout_template = {
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

        layout = layout_template

    else:

        layout = merge(layout_template, layout)

    data = []

    for i, x in enumerate(xs):

        color = get_color(DATA_TYPE_TO_COLORSCALE["categorical"], i, n_x)

        name = x.name

        trace_template = {
            "legendgroup": i,
            "name": name,
            "showlegend": 1 < len(xs),
            "x": x,
        }

        data.append(
            {
                "yaxis": "y2",
                "type": "histogram",
                "histnorm": histnorm,
                "xbins": {"size": bin_size},
                "marker": {"color": color},
                **trace_template,
            }
        )

        if plot_rug:

            data.append(
                {
                    "showlegend": False,
                    "y": (i,) * len(x),
                    "text": x.index.to_numpy(),
                    "mode": "markers",
                    "marker": {"symbol": "line-ns-open", "color": color},
                    "hoverinfo": "x+text",
                    **trace_template,
                }
            )

    plot_plotly({"layout": layout, "data": data}, html_file_path=html_file_path)
