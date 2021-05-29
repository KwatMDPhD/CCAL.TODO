from inspect import stack


def print_stack_state():

    s = stack()[1]

    print(
        "@{}({})".format(
            s[3],
            ", ".join("{}={}".format(k, v) for k, v in s[0].f_locals.items()),
        )
    )


def cast_builtin(o):

    for b in [
        None,
        False,
        True,
    ]:

        if o is b or o == str(b):

            return b

    for t in [int, float]:

        try:

            return t(o)

        except ValueError:

            pass

    return o


def check_is_iterable(an):

    try:

        iter(an)

        return True

    except:

        return False
