def cast_object_to_builtin(object):

    if object == "None":

        return None

    elif object == "True":

        return True

    elif object == "False":

        return False

    for type_ in (int, float):

        try:

            return type_(object)

        except ValueError:

            pass

    return object
