from .support import cast_builtin


def _update_with_suffix(
    array_2d,
):

    (axis_0_size, axis_1_size) = array_2d.shape

    for index_0 in range(axis_0_size):

        for index_1 in range(axis_1_size):

            value = array_2d[index_0, index_1]

            if isinstance(value, str):

                array_2d[index_0, index_1] = cast_builtin(value.split(": ", 1)[1])
