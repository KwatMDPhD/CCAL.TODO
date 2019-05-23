from numpy import arange, linspace, meshgrid

from .make_colorscale_from_colors import make_colorscale_from_colors
from .normalize_nd_array import normalize_nd_array
from .pick_colors import pick_colors
from .plot_and_save import plot_and_save


def plot_bubble_map(
    dataframe_size,
    dataframe_color=None,
    marker_size_max=32,
    showscale=None,
    colorbar_x=None,
    title=None,
    xaxis_title=None,
    yaxis_title=None,
    html_file_path=None,
):

    if dataframe_color is None:

        dataframe_color = dataframe_size

    axis_template = {"zeroline": False}

    x, y = meshgrid(
        linspace(0, dataframe_size.shape[1] - 1, num=dataframe_size.shape[1]),
        linspace(0, dataframe_size.shape[0] - 1, num=dataframe_size.shape[0]),
    )

    plot_and_save(
        {
            "layout": {
                "width": max(640, marker_size_max * 2 * dataframe_size.shape[1]),
                "height": max(640, marker_size_max * 2 * dataframe_size.shape[0]),
                "title": {"text": title},
                "xaxis": {
                    "tickvals": arange(dataframe_size.shape[1]),
                    "ticktext": dataframe_size.columns,
                    "title": f"{xaxis_title} ({dataframe_size.shape[1]})",
                    **axis_template,
                },
                "yaxis": {
                    "tickvals": arange(dataframe_size.shape[0]),
                    "ticktext": dataframe_size.index[::-1],
                    "title": f"{yaxis_title} ({dataframe_size.shape[0]})",
                    **axis_template,
                },
            },
            "data": [
                {
                    "type": "scatter",
                    "x": x.ravel(),
                    "y": y.ravel()[::-1],
                    "text": dataframe_size.values.ravel(),
                    "mode": "markers",
                    "marker": {
                        "size": normalize_nd_array(
                            dataframe_size.values, None, "0-1"
                        ).ravel()
                        * marker_size_max,
                        "color": dataframe_color.values.ravel(),
                        "colorscale": make_colorscale_from_colors(
                            pick_colors(dataframe_color)
                        ),
                        "showscale": True,
                        "colorbar": {
                            "x": colorbar_x,
                            "len": 0.64,
                            "thickness": marker_size_max / 3,
                        },
                        "line": {"color": "#000000"},
                    },
                }
            ],
        },
        html_file_path,
    )
