from numpy import apply_along_axis

from .array import normalize
from .clustering import cluster
from .CONSTANT import GOLDEN_FACTOR
from .plot import plot_heat_map, plot_plotly


def plot(
    w_,
    h_,
    axis_0_label__,
    axis_1_label__,
    axis_0_name_,
    axis_1_name_,
    error__,
    axis_factor_size=640,
    directory_path=None,
):

    axis_size = axis_factor_size * GOLDEN_FACTOR

    factor_axis = {"dtick": 1}

    for (w_index, w) in enumerate(w_):

        w = apply_along_axis(normalize, 1, w[cluster(w)[0], :], "-0-")

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}w_{}.html".format(directory_path, w_index)

        plot_heat_map(
            w,
            axis_0_label__[w_index],
            make_factor_label_(w.shape[1]),
            axis_0_name_[w_index],
            "Factor",
            layout={
                "height": axis_size,
                "width": axis_factor_size,
                "title": {"text": "W {}".format(w_index)},
                "xaxis": factor_axis,
            },
            file_path=file_path,
        )

    for (h_index, h) in enumerate(h_):

        h = apply_along_axis(normalize, 0, h[:, cluster(h.T)[0]], "-0-")

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}h_{}.html".format(directory_path, w_index)

        plot_heat_map(
            h,
            make_factor_label_(h.shape[0]),
            axis_1_label__[h_index],
            "Factor",
            axis_1_name_[h_index],
            layout={
                "height": axis_factor_size,
                "width": axis_size,
                "title": {"text": "H {}".format(h_index)},
                "yaxis": factor_axis,
            },
            file_path=file_path,
        )

    if directory_path is None:

        file_path = None

    else:

        file_path = "{}error.html".format(directory_path)

    plot_plotly(
        {
            "data": [
                {"name": index, "y": error_} for index, error_ in enumerate(error__)
            ],
            "layout": {
                "xaxis": {"title": "Iteration"},
                "yaxis": {"title": "Error"},
                "annotations": [
                    {
                        "x": error_.size - 1,
                        "y": error_[-1],
                        "text": "{:.2e}".format(error_[-1]),
                    }
                    for error_ in error__
                ],
            },
        },
        file_path=file_path,
    )
