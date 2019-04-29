from .compute_information_coefficient_between_2_1d_arrays import (
    compute_information_coefficient_between_2_1d_arrays,
)


def compute_information_distance_between_2_1d_arrays(
    _1d_array_0, _1d_array_1, n_grid=24
):

    return (
        1
        - compute_information_coefficient_between_2_1d_arrays(
            _1d_array_0, _1d_array_1, n_grid=n_grid
        )
    ) / 2
