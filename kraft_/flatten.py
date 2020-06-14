def flatten(iterable, types=(tuple, list, set)):

    list_ = list(iterable)

    i = 0

    while i < len(list_):

        while isinstance(list_[i], types):

            if len(list_[i]) != 0:

                list_[i : i + 1] = list_[i]

            else:

                list_.pop(i)

                i -= 1

                break

        i += 1

    return list_
