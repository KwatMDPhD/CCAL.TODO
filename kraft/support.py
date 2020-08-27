from inspect import stack


def print_function_information():

    stack_1 = stack()[1]

    try:

        argument_ = (
            "{} = {}".format(key, value)
            for key, value in sorted(stack_1[0].f_locals.items())
        )

        separater = "\n    "

        print("@ {}{}{}".format(stack_1[3], separater, separater.join(argument_)))

    finally:

        del stack_1


def cast_builtin(object):

    for builtin_object in (
        None,
        False,
        True,
    ):

        if object is builtin_object or object == str(builtin_object):

            return builtin_object

    for type in (float, int):

        try:

            return type(object)

        except ValueError:

            pass

    return object
