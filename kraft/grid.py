from numpy import asarray, diff, linspace, meshgrid, unique

from .plot import plot_heat_map, plot_plotly


def make_1d(min, max, extension_fracton, size):

    extension = (max - min) * extension_fracton

    min -= extension

    max += extension

    return linspace(min, max, num=size)


def reflect_1d(_1d_grid, reflecting_grid_number):

    _1d_grid_reflecting = _1d_grid.copy()

    for index, number in enumerate(_1d_grid):

        if number < reflecting_grid_number:

            _1d_grid_reflecting[index] += (reflecting_grid_number - number) * 2

        else:

            _1d_grid_reflecting[index] -= (number - reflecting_grid_number) * 2

    return _1d_grid_reflecting


def get_d(_1d_grid):

    return diff(unique(_1d_grid)).min()


def make_nd(_1d_grid_):

    return asarray(
        tuple(
            dimension_meshgrid.ravel()
            for dimension_meshgrid in meshgrid(*_1d_grid_, indexing="ij")
        )
    ).T


def get_1d(point_x_dimension):

    return tuple(unique(dimension) for dimension in point_x_dimension.T)


def plot(
    nd_grid, nd_vector, dimension_name_=None, number_name="Number", file_path=None,
):

    dimension_n = nd_grid.shape[1]

    if dimension_name_ is None:

        dimension_name_ = tuple(
            "Dimension {}".format(index) for index in range(dimension_n)
        )

    _1d_grid_ = get_1d(nd_grid)

    nd_number_array = nd_vector.reshape(tuple(_1d_grid.size for _1d_grid in _1d_grid_))

    for index, _1d_grid in enumerate(_1d_grid_):

        print(
            "Grid {}: size={} min={:.2e} max={:.2e}".format(
                index, _1d_grid.size, _1d_grid.min(), _1d_grid.max()
            )
        )

    print(
        "Number: min={:.2e} max={:.2e}".format(
            nd_number_array.min(), nd_number_array.max()
        )
    )

    if dimension_n == 1:

        plot_plotly(
            {
                "data": [{"y": nd_number_array, "x": _1d_grid_[0]}],
                "layout": {
                    "yaxis": {"title": {"text": number_name}},
                    "xaxis": {"title": {"text": dimension_name_[0]}},
                },
            },
            file_path=file_path,
        )

    elif dimension_n == 2:

        plot_heat_map(
            nd_number_array,
            asarray(tuple("{:.2e} *".format(number) for number in _1d_grid_[0])),
            asarray(tuple("* {:.2e}".format(number) for number in _1d_grid_[1])),
            dimension_name_[0],
            dimension_name_[1],
            layout={"title": {"text": number_name}},
            file_path=file_path,
        )
