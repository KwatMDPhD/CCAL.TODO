def flatten_nested_iterable(iterable, iterable_types=(tuple, list, set)):

    flattened_iterable = list(iterable)

    i = 0

    while i < len(flattened_iterable):

        while isinstance(flattened_iterable[i], iterable_types):

            if len(flattened_iterable[i]) == 0:

                flattened_iterable.pop(i)

                i -= 1

                break

            else:

                flattened_iterable[i : i + 1] = flattened_iterable[i]

        i += 1

    return flattened_iterable
