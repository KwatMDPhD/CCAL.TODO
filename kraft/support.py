from inspect import stack


def cast_builtin(object_):

    for builtin_object in (
        None,
        False,
        True,
    ):

        if object_ is builtin_object or object_ == str(builtin_object):

            return builtin_object

    for type_ in (int, float):

        try:

            return type_(object_)

        except ValueError:

            pass

    return object_


def print_function_information():

    stack_1 = stack()[1]

    try:

        arguments = (
            "{} = {}".format(key, value)
            for key, value in sorted(stack_1[0].f_locals.items())
        )

        separater = "\n    "

        print("@ {}{}{}".format(stack_1[3], separater, separater.join(arguments)))

    finally:

        del stack_1
