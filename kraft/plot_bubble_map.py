from numpy import arange, meshgrid

from .COLORBAR import COLORBAR
from .DATA_TYPE_COLORSCALE import DATA_TYPE_COLORSCALE
from .merge_2_dicts import merge_2_dicts
from .normalize import normalize
from .plot_plotly import plot_plotly


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
