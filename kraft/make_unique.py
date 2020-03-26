def make_unique(iterable):

    iterable_unique = []

    for x in iterable:

        x_original = x

        i = 2

        while x in iterable_unique:

            x = "{}.{}".format(x_original, i)

            i += 1

        iterable_unique.append(x)

    return iterable_unique
