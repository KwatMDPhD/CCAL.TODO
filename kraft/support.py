from inspect import stack


def print_function_information():

    s1 = stack()[1]

    print(
        "@{}({})".format(
            s1[3],
            ", ".join("{}={}".format(k, v) for k, v in s1[0].f_locals.items()),
        )
    )


def cast_builtin(o):

    for bo in [
        None,
        False,
        True,
    ]:

        if o is bo or o == str(bo):

            return bo

    for t in [int, float]:

        try:

            return t(o)

        except ValueError:

            pass

    return o
