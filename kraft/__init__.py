from .array import (
    clip,
    error_nan,
    guess_type,
    ignore_nan_and_function_1,
    ignore_nan_and_function_2,
    is_sorted,
    log,
    normalize,
    shift_minimum,
)
from .plot import get_color, plot_bubble_map, plot_heat_map, plot_histogram, plot_plotly
from .point_x_dimension import (
    get_grids,
    grid,
    make_grid_point_x_dimension,
    plot_grid_point_x_dimension,
)
from .support import cast_builtin, merge_2_dicts
