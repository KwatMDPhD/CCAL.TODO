def cast_object_to_builtin(object_):

    if object_ == "None":

        return None

    elif object_ == "True":

        return True

    elif object_ == "False":

        return False

    for type_ in (int, float):

        try:

            return type_(object_)

        except ValueError:

            pass

    return object_
