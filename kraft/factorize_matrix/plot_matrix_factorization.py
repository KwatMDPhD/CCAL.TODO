from numpy import apply_along_axis

from .cluster import cluster
from .normalize import normalize
from .plot_heat_map import plot_heat_map
from .plot_plotly import plot_plotly


def plot_matrix_factorization(ws, hs, errors=None, axis_size=320, directory_path=None):

    axis_size_ = axis_size * 1.618

    for w_index, w in enumerate(ws):

        w = apply_along_axis(normalize, 1, w[cluster(w)[0], :], "-0-")

        layout_factor_axis = {"title": {"text": "Factor"}, "dtick": 1}

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = "{}/w{}.html".format(directory_path, w_index)

        plot_heat_map(
            w,
            layout={
                "height": axis_size_,
                "width": axis_size,
                "title": {"text": "W{}".format(w_index)},
                "xaxis": layout_factor_axis,
            },
            html_file_path=html_file_path,
        )

    for h_index, h in enumerate(hs):

        h = apply_along_axis(normalize, 0, h[:, cluster(h.T)[0]], "-0-")

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = "{}/h{}.html".format(directory_path, w_index)

        plot_heat_map(
            h,
            layout={
                "height": axis_size,
                "width": axis_size_,
                "title": {"text": "H{}".format(h_index)},
                "yaxis": layout_factor_axis,
            },
            html_file_path=html_file_path,
        )

    if errors is not None:

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = "{}/error.html".format(directory_path)

        plot_plotly(
            {
                "layout": {
                    "xaxis": {"title": "Iteration"},
                    "yaxis": {"title": "Error"},
                    "annotations": [
                        {
                            "x": error.size - 1,
                            "y": error[-1],
                            "text": "{:.2e}".format(error[-1]),
                        }
                        for error in errors
                    ],
                },
                "data": [
                    {"name": error_index, "y": error}
                    for error_index, error in enumerate(errors)
                ],
            },
            html_file_path=html_file_path,
        )
