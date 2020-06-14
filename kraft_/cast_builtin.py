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
