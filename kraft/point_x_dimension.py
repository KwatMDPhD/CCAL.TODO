from numpy import asarray, linspace, meshgrid, unique
from pandas import DataFrame, Index

from .plot import plot_heat_map, plot_plotly


def grid(min_, max_, fraction_extension, n):

    assert 0 <= fraction_extension

    extension = (max_ - min_) * fraction_extension

    min_ -= extension

    max_ += extension

    return linspace(min_, max_, num=n)


def make_grid_point_x_dimension(grids):

    return asarray(tuple(array.ravel() for array in meshgrid(*grids, indexing="ij"))).T


def get_grids(point_x_dimension):

    return tuple(unique(dimension) for dimension in point_x_dimension.T)


def plot_grid_point_x_dimension(
    grid_point_x_dimension,
    grid_point_value,
    names=None,
    value_name="Value",
    html_file_path=None,
):

    n_dimension = grid_point_x_dimension.shape[1]

    if names is None:

        names = tuple("Dimension {}".format(i) for i in range(n_dimension))

    grids = get_grids(grid_point_x_dimension)

    grid_point_value = grid_point_value.reshape(tuple(grid.size for grid in grids))

    for grid_index, grid in enumerate(grids):

        print(
            "Grid {}: size={} min={:.2e} max={:.2e}".format(
                grid_index, grid.size, grid.min(), grid.max()
            )
        )

    print(
        "Value: min={:.2e} max={:.2e}".format(
            grid_point_value.min(), grid_point_value.max()
        )
    )

    if n_dimension == 1:

        plot_plotly(
            {
                "layout": {
                    "xaxis": {"title": {"text": names[0]}},
                    "yaxis": {"title": {"text": value_name}},
                },
                "data": [
                    {
                        "type": "scatter",
                        "x": grids[0],
                        "y": grid_point_value,
                        "mode": "lines",
                    }
                ],
            },
            html_file_path=html_file_path,
        )

    elif n_dimension == 2:

        plot_heat_map(
            DataFrame(
                grid_point_value,
                index=Index(("{:.2e} *".format(i) for i in grids[0]), name=names[0]),
                columns=Index(("* {:.2e}".format(i) for i in grids[1]), name=names[1]),
            ),
            layout={"title": {"text": value_name}},
            html_file_path=html_file_path,
        )
