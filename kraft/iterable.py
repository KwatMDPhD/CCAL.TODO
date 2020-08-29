def flatten(iterable, type_=(tuple, list, set)):

    list_ = list(iterable)

    index = 0

    while index < len(list_):

        while isinstance(list_[index], type_):

            if len(list_[index]) == 0:

                list_.pop(index)

                index -= 1

                break

            else:

                list_[index : index + 1] = list_[index]

        index += 1

    return tuple(list_)
