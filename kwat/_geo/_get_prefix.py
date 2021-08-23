def _get_prefix(
    row,
):

    return tuple(
        set(value.split(": ", 1)[0] for value in row if isinstance(value, str))
    )
