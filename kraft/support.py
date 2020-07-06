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

    frame_info = stack()[1]

    try:

        arguments = (
            "{} = {}".format(key, value)
            for key, value in sorted(frame_info[0].f_locals.items())
        )

        separater = "\n    "

        print("@ {}{}{}".format(frame_info[3], separater, separater.join(arguments)))

    finally:

        del frame_info
