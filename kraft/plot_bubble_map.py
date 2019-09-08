from numpy import arange, meshgrid

from .make_colorscale_from_colors import make_colorscale_from_colors
from .normalize_array import normalize_array
from .pick_colors import pick_colors
from .plot_plotly_figure import plot_plotly_figure


def plot_bubble_map(
    dataframe_size,
    dataframe_color=None,
    marker_size_max=32,
    colorbar=None,
    layout=None,
    xaxis=None,
    yaxis=None,
    html_file_path=None,
):

    x_grid = arange(dataframe_size.shape[1])

    y_grid = arange(dataframe_size.shape[0])

    mesh_grid_x, mesh_grid_y = meshgrid(x_grid, y_grid)

    xaxis_template = {
        "zeroline": False,
        "tickvals": x_grid,
        "ticktext": dataframe_size.columns,
    }

    if xaxis is None:

        xaxis = xaxis_template

    else:

        xaxis = {**xaxis_template, **xaxis}

    yaxis_template = {
        "zeroline": False,
        "tickvals": y_grid,
        "ticktext": dataframe_size.index,
    }

    if yaxis is None:

        yaxis = yaxis_template

    else:

        yaxis = {**yaxis_template, **yaxis}

    layout_template = {
        "height": max(640, marker_size_max * 2 * y_grid.size),
        "width": max(640, marker_size_max * 2 * x_grid.size),
        "xaxis": xaxis,
        "yaxis": yaxis,
    }

    if layout is None:

        layout = layout_template

    else:

        layout = {**layout_template, **layout}

    if dataframe_color is None:

        dataframe_color = dataframe_size

    plot_plotly_figure(
        {
            "layout": layout,
            "data": [
                {
                    "type": "scatter",
                    "x": mesh_grid_x.ravel(),
                    "y": mesh_grid_y.ravel()[::-1],
                    "text": dataframe_size.values.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize_array(
                            dataframe_size.values, None, "0-1"
                        ).ravel()
                        * marker_size_max,
                        "color": dataframe_color.values.ravel(),
                        "colorscale": make_colorscale_from_colors(
                            pick_colors(dataframe_color)
                        ),
                        "showscale": True,
                        "colorbar": colorbar,
                        "line": {"color": "#000000"},
                    },
                }
            ],
        },
        html_file_path,
    )
