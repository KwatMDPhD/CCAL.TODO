from numpy import asarray, linspace, meshgrid, unique
from pandas import DataFrame, Index

from .plot import plot_heat_map, plot_plotly


def make_grid_1d(min_, max_, fraction_extension, size):

    assert 0 <= fraction_extension

    extension = (max_ - min_) * fraction_extension

    min_ -= extension

    max_ += extension

    return linspace(min_, max_, num=size)


def make_grid_1d_for_reflecting(grid_1d, grid_number_for_reflecting):

    grid_1d_for_reflection = grid_1d.copy()

    for i, number in enumerate(grid_1d):

        if number < grid_number_for_reflecting:

            grid_1d_for_reflection[i] += (grid_number_for_reflecting - number) * 2

        else:

            grid_1d_for_reflection[i] -= (number - grid_number_for_reflecting) * 2

    return grid_1d_for_reflection


def make_grid_nd(grid_1ds):

    return asarray(
        tuple(
            dimension_meshgrid.ravel()
            for dimension_meshgrid in meshgrid(*grid_1ds, indexing="ij")
        )
    ).T


def get_grid_1ds(point_x_dimension):

    return tuple(unique(dimension) for dimension in point_x_dimension.T)


def shape(grid_nd_numbers, grid_1ds):

    return grid_nd_numbers.reshape(tuple(grid_1d.size for grid_1d in grid_1ds))


def plot_grid_nd(
    grid_nd,
    grid_nd_numbers,
    dimension_names=None,
    number_name="Number",
    html_file_path=None,
):

    n_dimension = grid_nd.shape[1]

    if dimension_names is None:

        dimension_names = tuple("Dimension {}".format(i) for i in range(n_dimension))

    grid_1ds = get_grid_1ds(grid_nd)

    grid_nd_numbers_shape = shape(grid_nd_numbers, grid_1ds)

    for i, grid_1d in enumerate(grid_1ds):

        print(
            "Grid {}: size={} min={:.2e} max={:.2e}".format(
                i, grid_1d.size, grid_1d.min(), grid_1d.max()
            )
        )

    print(
        "Number: min={:.2e} max={:.2e}".format(
            grid_nd_numbers_shape.min(), grid_nd_numbers_shape.max()
        )
    )

    if n_dimension == 1:

        plot_plotly(
            {
                "layout": {
                    "xaxis": {"title": {"text": dimension_names[0]}},
                    "yaxis": {"title": {"text": number_name}},
                },
                "data": [{"x": grid_1ds[0], "y": grid_nd_numbers_shape}],
            },
            html_file_path=html_file_path,
        )

    elif n_dimension == 2:

        plot_heat_map(
            DataFrame(
                grid_nd_numbers_shape,
                index=Index(
                    ("{:.2e} *".format(number) for number in grid_1ds[0]),
                    name=dimension_names[0],
                ),
                columns=Index(
                    ("* {:.2e}".format(number) for number in grid_1ds[1]),
                    name=dimension_names[1],
                ),
            ),
            layout={"title": {"text": number_name}},
            html_file_path=html_file_path,
        )
