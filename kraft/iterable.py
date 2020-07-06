def flatten(iterable, types=(tuple, list, set)):

    iterable_flat = list(iterable)

    i = 0

    while i < len(iterable_flat):

        while isinstance(iterable_flat[i], types):

            if len(iterable_flat[i]) == 0:

                iterable_flat.pop(i)

                i -= 1

                break

            else:

                iterable_flat[i : i + 1] = iterable_flat[i]

        i += 1

    return tuple(iterable_flat)


def map_int(iterable):

    object_to_i = {}

    i_to_object = {}

    i = 0

    for object_ in iterable:

        if object_ not in object_to_i:

            object_to_i[object_] = i

            i_to_object[i] = object_

            i += 1

    return object_to_i, i_to_object


def make_unique(strs):

    strs_unique = []

    for str_ in strs:

        x_original = str_

        i = 2

        while str_ in strs_unique:

            str_ = "{}{}".format(x_original, i)

            i += 1

        strs_unique.append(str_)

    return strs_unique
